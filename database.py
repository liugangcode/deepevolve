# Adapted from OpenEvolve

import json
import logging
import os
import random
import time
from rapidfuzz.distance import Levenshtein

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

from utils.datatypes import IdeaData

logger = logging.getLogger(__name__)

class Program(BaseModel):
    """Represents a program in the database"""

    # Program identification
    id: str
    code: str
    idea: IdeaData

    # Evolution information
    timestamp: float = Field(default_factory=time.time)
    parent_id: Optional[str] = None
    evolution_history: List[IdeaData] = Field(
        default_factory=list
    )  # Track the idea evolution history of the program
    iteration_found: int = 0  # Track which iteration this program was found
    report: Optional[str] = None  # Track the LLM report of the program

    # Performance metrics
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    language: str = "python"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation with 'report' second-to-last and 'code' last"""
        base_dict = self.dict()
        # Remove and reorder specific fields
        code_value = base_dict.pop("code", None)
        report_value = base_dict.pop("report", None)

        ordered = {k: base_dict[k] for k in base_dict if k not in ("report", "code")}
        ordered["report"] = report_value
        ordered["code"] = code_value
        return ordered

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation"""
        return cls(**data)


class ProgramDatabase:
    """
    Database for storing and sampling programs during evolution

    The database implements a combination of MAP-Elites algorithm and
    island-based population model to maintain diversity during evolution.
    It also tracks the absolute best program separately to ensure it's never lost.
    """

    def __init__(self, config):
        self.config = config

        # In-memory program storage
        self.programs: Dict[str, Program] = {}

        # Feature grid for MAP-Elites
        self.feature_map: Dict[str, str] = {}
        self.feature_bins = config.feature_bins

        # Island populations
        self.islands: List[Set[str]] = [set() for _ in range(config.num_islands)]

        # Island-based evolution tracking
        self.current_island: int = 0  # Track which island we're currently evolving
        self.island_generations: List[int] = [0] * config.num_islands

        # Migration parameters
        self.migration_interval: int = getattr(config, "migration_interval", 10)
        self.migration_rate: float = getattr(config, "migration_rate", 0.1)
        self.last_migration_generation: int = 0

        # Archive of elite programs
        self.archive: Set[str] = set()

        # Track the absolute best program separately
        self.best_program_id: Optional[str] = None

        # Track the last iteration number (for resuming)
        self.last_iteration: int = 0

        # Load database from disk if path is provided
        if config.db_path and os.path.exists(config.db_path):
            self.load(config.db_path)

        # Set random seed for reproducible sampling if specified
        # if config.random_seed is not None:
        if config.random_seed is not None:
            import random

            random.seed(config.random_seed)
            logger.info(f"Database: Set random seed to {config.random_seed}")

        self._distance_cache: Dict[Tuple[str, str], float] = {}
        self.diversity_bin: Dict[str, int] = {}

        logger.info(f"Initialized program database with {len(self.programs)} programs")

    def add(
        self,
        program: Program,
        iteration: int = None,
        target_island: Optional[int] = None,
    ) -> str:
        """
        Add a program to the database

        Args:
            program: Program to add
            iteration: Current iteration (defaults to last_iteration)
            target_island: Specific island to add to (uses current_island if None)

        Returns:
            Program ID
        """
        # Store the program
        # If iteration is provided, update the program's iteration_found
        if iteration is not None:
            program.iteration_found = iteration
            # Update last_iteration if needed
            self.last_iteration = max(self.last_iteration, iteration)

        self.programs[program.id] = program

        # Calculate feature coordinates for MAP-Elites
        self.feature_map = {}
        for each_program in self.programs.values():
            # since we use min-max norm from all programs, we need to update the feature map for each program
            self._update_global_diversity()
            feature_coords = self._calculate_feature_coords(each_program)
            feature_key = self._feature_coords_to_key(feature_coords)
            if feature_key not in self.feature_map or self._is_better(
                each_program, self.programs[self.feature_map[feature_key]]
            ):
                self.feature_map[feature_key] = each_program.id

        # Add to specific island (not random!)
        island_idx = target_island if target_island is not None else self.current_island
        island_idx = island_idx % len(self.islands)  # Ensure valid island
        self.islands[island_idx].add(program.id)

        # Track which island this program belongs to
        program.metadata["island"] = island_idx

        # Update archive
        self._update_archive(program)

        # Update the absolute best program tracking
        self._update_best_program(program)

        # Save to disk if configured
        if self.config.db_path:
            self._save_program(program)

        logger.info(f"Added program {program.id} to island {island_idx}")
        
        # Enforce population size limit
        self._enforce_population_limit()
        return program.id

    def get(self, program_id: str) -> Optional[Program]:
        """
        Get a program by ID

        Args:
            program_id: Program ID

        Returns:
            Program or None if not found
        """
        return self.programs.get(program_id)

    def sample(self) -> Tuple[Program, List[Program]]:
        """
        Sample a program and inspirations for the next evolution step

        Returns:
            Tuple of (parent_program, inspiration_programs)
        """
        # Select parent program
        parent = self._sample_parent()

        # Select inspirations
        inspirations = self._sample_inspirations(parent, n=self.config.n_inspirations)

        logger.info(f"Sampled parent {parent.id} and {len(inspirations)} inspirations")
        return parent, inspirations

    def get_best_program(self, metric: str = 'combined_score') -> Optional[Program]:
        """
        Get the best program based on a metric

        Args:
            metric: Metric to use for ranking (uses combined_score as default)

        Returns:
            Best program or None if database is empty
        """
        if not self.programs:
            return None

        # If no specific metric and we have a tracked best program, return it
        if (
            metric is None
            and self.best_program_id
            and self.best_program_id in self.programs
        ):
            logger.info(f"Using tracked best program: {self.best_program_id}")
            return self.programs[self.best_program_id]

        if metric:
            # Sort by specific metric
            sorted_programs = sorted(
                [p for p in self.programs.values() if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
            if sorted_programs:
                logger.info(
                    f"Found best program by metric '{metric}': {sorted_programs[0].id}"
                )
        elif self.programs and all(
            "combined_score" in p.metrics for p in self.programs.values()
        ):
            # Sort by combined_score if it exists (preferred method)
            sorted_programs = sorted(
                self.programs.values(),
                key=lambda p: p.metrics["combined_score"],
                reverse=True,
            )
            if sorted_programs:
                logger.info(
                    f"Found best program by combined_score: {sorted_programs[0].id}"
                )
        else:
            raise ValueError("No metric provided to get the best program. Using combined_score as default.")
        
        # Update the best program tracking if we found a better program
        if sorted_programs and (
            self.best_program_id is None
            or sorted_programs[0].id != self.best_program_id
        ):
            old_id = self.best_program_id
            self.best_program_id = sorted_programs[0].id
            logger.info(
                f"Updated best program tracking from {old_id} to {self.best_program_id}"
            )

            # Also log the scores to help understand the update
            if (
                old_id
                and old_id in self.programs
                and "combined_score" in self.programs[old_id].metrics
                and "combined_score" in self.programs[self.best_program_id].metrics
            ):
                old_score = self.programs[old_id].metrics["combined_score"]
                new_score = self.programs[self.best_program_id].metrics[
                    "combined_score"
                ]
                logger.info(
                    f"Score change: {old_score:.4f} → {new_score:.4f} ({new_score-old_score:+.4f})"
                )

        return sorted_programs[0] if sorted_programs else None

    def get_top_programs(
        self, n: int = 10, metric: str = 'combined_score'
    ) -> List[Program]:
        """
        Get the top N programs based on a metric

        Args:
            n: Number of programs to return
            metric: Metric to use for ranking (uses combined_score as default)

        Returns:
            List of top programs
        """
        if not self.programs:
            return []

        sorted_programs = sorted(
            [p for p in self.programs.values() if metric in p.metrics],
            key=lambda p: p.metrics[metric],
            reverse=True,
        )

        return sorted_programs[:n]

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        """
        Save the database to disk

        Args:
            path: Path to save to (uses config.db_path if None)
            iteration: Current iteration number
        """
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save each program
        for program in self.programs.values():
            self._save_program(program, save_path)

        # Save metadata
        metadata = {
            "feature_map": self.feature_map,
            "islands": [list(island) for island in self.islands],
            "archive": list(self.archive),
            "best_program_id": self.best_program_id,
            "last_iteration": iteration or self.last_iteration,
            "current_island": self.current_island,
            "island_generations": self.island_generations,
            "last_migration_generation": self.last_migration_generation,
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved database with {len(self.programs)} programs to {save_path}")

    def load(self, path: str) -> None:
        """
        Load the database from disk

        Args:
            path: Path to load from
        """
        if not os.path.exists(path):
            logger.warning(f"Database path {path} does not exist, skipping load")
            return

        # Load metadata
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.feature_map = metadata.get("feature_map", {})
            self.islands = [set(island) for island in metadata.get("islands", [])]
            self.archive = set(metadata.get("archive", []))
            self.best_program_id = metadata.get("best_program_id")
            self.last_iteration = metadata.get("last_iteration", 0)
            self.current_island = metadata.get("current_island", 0)
            self.island_generations = metadata.get(
                "island_generations", [0] * len(self.islands)
            )
            self.last_migration_generation = metadata.get(
                "last_migration_generation", 0
            )

            # Ensure island_generations list has correct length
            if len(self.island_generations) != len(self.islands):
                self.island_generations = [0] * len(self.islands)

            logger.info(
                f"Loaded database metadata with last_iteration={self.last_iteration}"
            )

        # Load programs
        programs_dir = os.path.join(path, "programs")
        if os.path.exists(programs_dir):
            for program_file in os.listdir(programs_dir):
                if program_file.endswith(".json"):
                    program_path = os.path.join(programs_dir, program_file)
                    try:
                        with open(program_path, "r") as f:
                            program_data = json.load(f)

                        program = Program.from_dict(program_data)
                        self.programs[program.id] = program
                    except Exception as e:
                        logger.warning(
                            f"Error loading program {program_file}: {str(e)}"
                        )

        logger.info(f"Loaded database with {len(self.programs)} programs from {path}")

    def _save_program(self, program: Program, base_path: Optional[str] = None) -> None:
        """
        Save a program to disk

        Args:
            program: Program to save
            base_path: Base path to save to (uses config.db_path if None)
        """
        save_path = base_path or self.config.db_path
        if not save_path:
            return

        # Create programs directory if it doesn't exist
        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)

        # Save program
        program_path = os.path.join(programs_dir, f"{program.id}.json")
        with open(program_path, "w") as f:
            json.dump(program.to_dict(), f, indent=2)

    def _update_global_diversity(self) -> float:
        """
        Update the global diversity for a program
        """
        avg_distances: Dict[str, float] = {}
        ids = list(self.programs.keys())

        # Compute each program's average distance to all others
        for pid1 in ids:
            code1 = self.programs[pid1].code
            total = 0.0
            count = 0

            for pid2 in ids:
                if pid1 == pid2:
                    continue

                # Use a sorted tuple as cache key
                key = (pid1, pid2) if pid1 < pid2 else (pid2, pid1)
                if key in self._distance_cache:
                    d = self._distance_cache[key]
                else:
                    d = Levenshtein.distance(code1, self.programs[pid2].code)
                    self._distance_cache[key] = d

                total += d
                count += 1

            avg_distances[pid1] = (total / count) if count > 0 else 0.0

        # Find global min and max of those averages
        all_avgs = list(avg_distances.values())
        if all_avgs:
            min_avg = min(all_avgs)
            max_avg = max(all_avgs)
        else:
            min_avg = max_avg = 0.0

        span = (max_avg - min_avg) if (max_avg > min_avg) else 1.0

        # Build a dict: program_id → normalized bin index
        self.diversity_bin: Dict[str, int] = {}
        for pid, avg in avg_distances.items():
            norm = (avg - min_avg) / span
            idx = int(norm * self.feature_bins)
            if idx >= self.feature_bins:
                idx = self.feature_bins - 1
            self.diversity_bin[pid] = idx

        return self.diversity_bin

    def _calculate_feature_coords(self, program: Program) -> List[int]:
        """
        Calculate feature coordinates for the MAP-Elites grid

        Args:
            program: Program to calculate features for

        Returns:
            List of feature coordinates
        """
        coords: List[int] = []
        for dim in self.config.feature_dimensions:
            if dim == "complexity":
                # Min-max normalize code length
                all_lens = [len(p.code) for p in self.programs.values()]
                min_len = min(all_lens)
                max_len = max(all_lens)
                span_len = (max_len - min_len + 1) if max_len > min_len else 1

                rel = (len(program.code) - min_len) / span_len
                idx = int(rel * self.feature_bins)
                if idx >= self.feature_bins:
                    idx = self.feature_bins - 1
                coords.append(idx)

            elif dim == "diversity":
                if program.id in self.diversity_bin:
                    coords.append(self.diversity_bin[program.id])
                else:
                    logger.warning(f"Program {program.id} not found in diversity bin but diversity is used as a feature")
                    coords.append(0)

            elif dim == "score":
                # Min-max normalize combined_score across all programs
                all_scores = []
                for p in self.programs.values():
                    if "combined_score" in p.metrics:
                        all_scores.append(p.metrics["combined_score"])
                current = program.metrics.get("combined_score", 0.0)
                all_scores.append(current)

                if all_scores:
                    min_score = min(all_scores)
                    max_score = max(all_scores) + 1e-8
                    norm_score = (current - min_score) / (max_score - min_score)
                    idx = int(norm_score * self.feature_bins)
                    if idx >= self.feature_bins:
                        idx = self.feature_bins - 1
                else:
                    idx = 0
                coords.append(idx)

            elif dim in program.metrics:
                # Normalize a specific metric
                score = program.metrics[dim]
                idx = int(score * self.feature_bins)
                if idx >= self.feature_bins:
                    idx = self.feature_bins - 1
                coords.append(idx)

            else:
                # Default to middle bin
                coords.append(self.feature_bins // 2)

        return coords

    def _feature_coords_to_key(self, coords: List[int]) -> str:
        """
        Convert feature coordinates to a string key

        Args:
            coords: Feature coordinates

        Returns:
            String key
        """
        return "-".join(str(c) for c in coords)

    def _is_better(self, program1: Program, program2: Program) -> bool:
        """
        Determine if program1 is better than program2

        Args:
            program1: First program
            program2: Second program

        Returns:
            True if program1 is better than program2
        """
        # If no metrics, use newest
        if not program1.metrics and not program2.metrics:
            return program1.timestamp > program2.timestamp

        # If only one has metrics, it's better
        if program1.metrics and not program2.metrics:
            return True
        if not program1.metrics and program2.metrics:
            return False

        # Check for combined_score first (this is the preferred metric)
        if (
            "combined_score" in program1.metrics
            and "combined_score" in program2.metrics
        ):
            return (
                program1.metrics["combined_score"] > program2.metrics["combined_score"]
            )

        # Fallback to average of all metrics
        logger.warning(f"Fallback to average of all metrics, which assume all higher is better: {program1.metrics} {program2.metrics}")
        avg1 = sum(program1.metrics.values()) / len(program1.metrics)
        avg2 = sum(program2.metrics.values()) / len(program2.metrics)

        return avg1 > avg2

    def _update_archive(self, program: Program) -> None:
        """
        Update the archive of elite programs

        Args:
            program: Program to consider for archive
        """
        # If archive not full, add program
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return

        # Otherwise, find worst program in archive
        archive_programs = [self.programs[pid] for pid in self.archive]
        worst_program = min(
            archive_programs,
            key=lambda p: p.metrics['combined_score'],
        )

        # Replace if new program is better
        if self._is_better(program, worst_program):
            self.archive.remove(worst_program.id)
            self.archive.add(program.id)

    def _update_best_program(self, program: Program) -> None:
        """
        Update the absolute best program tracking

        Args:
            program: Program to consider as the new best
        """
        # If we don't have a best program yet, this becomes the best
        if self.best_program_id is None:
            self.best_program_id = program.id
            logger.info(f"Set initial best program to {program.id}")
            return

        # Compare with current best program
        current_best = self.programs[self.best_program_id]

        # Update if the new program is better
        if self._is_better(program, current_best):
            old_id = self.best_program_id
            self.best_program_id = program.id

            # Log the change
            if (
                "combined_score" in program.metrics
                and "combined_score" in current_best.metrics
            ):
                old_score = current_best.metrics["combined_score"]
                new_score = program.metrics["combined_score"]
                score_diff = new_score - old_score
                logger.info(
                    f"New best program {program.id} replaces {old_id} (combined_score: {old_score:.4f} → {new_score:.4f}, +{score_diff:.4f})"
                )
            else:
                logger.info(f"New best program {program.id} replaces {old_id}")

    def _sample_parent(self) -> Program:
        """
        Sample a parent program from the current island for the next evolution step

        Returns:
            Parent program from current island
        """
        # Use exploration_ratio and exploitation_ratio to decide sampling strategy
        rand_val = random.random()

        if rand_val < self.config.exploration_ratio:
            # EXPLORATION: Sample from current island (diverse sampling)
            return self._sample_exploration_parent()
        elif rand_val < self.config.exploration_ratio + self.config.exploitation_ratio:
            # EXPLOITATION: Sample from archive (elite programs)
            return self._sample_exploitation_parent()
        else:
            # RANDOM: Sample from any program (remaining probability)
            return self._sample_random_parent()

    def _sample_exploration_parent(self) -> Program:
        """
        Sample a parent for exploration (from current island)
        """
        current_island_programs = self.islands[self.current_island]

        if not current_island_programs:
            # If current island is empty, initialize with best program or random program
            if self.best_program_id and self.best_program_id in self.programs:
                # Clone best program to current island
                best_program = self.programs[self.best_program_id]
                self.islands[self.current_island].add(self.best_program_id)
                best_program.metadata["island"] = self.current_island
                logger.info(
                    f"Initialized empty island {self.current_island} with best program"
                )
                return best_program
            else:
                # Use any available program
                return next(iter(self.programs.values()))

        # Sample from current island
        parent_id = random.choice(list(current_island_programs))
        return self.programs[parent_id]

    def _sample_exploitation_parent(self) -> Program:
        """
        Sample a parent for exploitation (from archive/elite programs)
        """
        if not self.archive:
            # Fallback to exploration if no archive
            return self._sample_exploration_parent()

        # Prefer programs from current island in archive
        archive_programs_in_island = [
            pid
            for pid in self.archive
            if pid in self.programs
            and self.programs[pid].metadata.get("island") == self.current_island
        ]

        if archive_programs_in_island:
            parent_id = random.choice(archive_programs_in_island)
            return self.programs[parent_id]
        else:
            # Fall back to any archive program if current island has none
            parent_id = random.choice(list(self.archive))
            return self.programs[parent_id]

    def _sample_random_parent(self) -> Program:
        """
        Sample a completely random parent from all programs
        """
        if not self.programs:
            raise ValueError("No programs available for sampling")

        # Sample randomly from all programs
        program_id = random.choice(list(self.programs.keys()))
        return self.programs[program_id]

    def _sample_inspirations(self, parent: Program, n: int = 5) -> List[Program]:
        """
        Sample inspiration programs for the next evolution step

        Args:
            parent: Parent program
            n: Number of inspirations to sample

        Returns:
            List of inspiration programs
        """
        inspirations = []

        # Always include the absolute best program if available and different from parent
        if self.best_program_id is not None and self.best_program_id != parent.id:
            best_program = self.programs[self.best_program_id]
            inspirations.append(best_program)
            logger.info(
                f"Including best program {self.best_program_id} in inspirations"
            )

        # Add top programs as inspirations
        top_n = max(1, int(n * self.config.elite_selection_ratio))
        top_programs = self.get_top_programs(n=top_n, metric='combined_score')
        for program in top_programs:
            if (
                program.id not in [p.id for p in inspirations]
                and program.id != parent.id
            ):
                inspirations.append(program)

        # Add diverse programs using config.num_diverse_programs
        if len(self.programs) > n and len(inspirations) < n:
            # Calculate how many diverse programs to add (up to remaining slots)
            remaining_slots = n - len(inspirations)

            # Sample from different feature cells for diversity
            feature_coords = self._calculate_feature_coords(parent)

            # Get programs from nearby feature cells
            nearby_programs = []
            for _ in range(remaining_slots):
                # Perturb coordinates
                perturbed_coords = [
                    max(0, min(self.feature_bins - 1, c + random.randint(-1, 1)))
                    for c in feature_coords
                ]

                # Try to get program from this cell
                cell_key = self._feature_coords_to_key(perturbed_coords)
                if cell_key in self.feature_map:
                    program_id = self.feature_map[cell_key]
                    if program_id != parent.id and program_id not in [
                        p.id for p in inspirations
                    ]:
                        nearby_programs.append(self.programs[program_id])

            # If we need more, add random programs
            if len(inspirations) + len(nearby_programs) < n:
                remaining = n - len(inspirations) - len(nearby_programs)
                all_ids = set(self.programs.keys())
                excluded_ids = (
                    {parent.id}
                    .union(p.id for p in inspirations)
                    .union(p.id for p in nearby_programs)
                )
                available_ids = list(all_ids - excluded_ids)

                if available_ids:
                    random_ids = random.sample(
                        available_ids, min(remaining, len(available_ids))
                    )
                    random_programs = [self.programs[pid] for pid in random_ids]
                    nearby_programs.extend(random_programs)

            inspirations.extend(nearby_programs)

        return inspirations[:n]

    def _enforce_population_limit(self) -> None:
        """
        Enforce the population size limit by removing worst programs if needed
        """
        if len(self.programs) <= self.config.population_size:
            return

        # Calculate how many programs to remove
        num_to_remove = len(self.programs) - self.config.population_size

        logger.info(
            f"Population size ({len(self.programs)}) exceeds limit ({self.config.population_size}), removing {num_to_remove} programs"
        )

        # Get programs sorted by fitness (worst first)
        all_programs = list(self.programs.values())

        # Sort by combined_score (higher better) metric (worst first)
        sorted_programs = sorted(
            all_programs,
            key=lambda p: p.metrics["combined_score"] if "combined_score" in p.metrics else 0.0,
        )

        # Remove worst programs, but never remove the best program
        programs_to_remove = []
        for program in sorted_programs:
            if len(programs_to_remove) >= num_to_remove:
                break
            # Don't remove the best program
            if program.id != self.best_program_id:
                programs_to_remove.append(program)

        # If we still need to remove more and only have the best program protected,
        # remove from the remaining programs anyway (but keep the absolute best)
        if len(programs_to_remove) < num_to_remove:
            remaining_programs = [
                p
                for p in sorted_programs
                if p not in programs_to_remove and p.id != self.best_program_id
            ]
            additional_removals = remaining_programs[
                : num_to_remove - len(programs_to_remove)
            ]
            programs_to_remove.extend(additional_removals)

        # Remove the selected programs
        for program in programs_to_remove:
            program_id = program.id

            # Remove from main programs dict
            if program_id in self.programs:
                del self.programs[program_id]
                logger.info(f"Removed program {program_id} from self.programs")

            # Remove from feature map
            keys_to_remove = []
            for key, pid in self.feature_map.items():
                if pid == program_id:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.feature_map[key]
                logger.info(f"Removed program {program_id} from self.feature_map")
            
            # Remove from islands
            for island_idx, island in enumerate(self.islands):
                island.discard(program_id)
                logger.info(f"Removed program {program_id} from self.islands[{island_idx}]")

            # Remove from archive
            self.archive.discard(program_id)
            logger.info(f"Removed program {program_id} from self.archive")

            logger.info(f"Removed program {program_id} due to population limit")

        logger.info(f"Population size after cleanup: {len(self.programs)}")

    # Island management methods
    def set_current_island(self, island_idx: int) -> None:
        """Set which island is currently being evolved"""
        self.current_island = island_idx % len(self.islands)
        logger.info(f"Switched to evolving island {self.current_island}")

    def next_island(self) -> int:
        """Move to the next island in round-robin fashion"""
        self.current_island = (self.current_island + 1) % len(self.islands)
        logger.info(f"Advanced to island {self.current_island}")
        return self.current_island

    def increment_island_generation(self, island_idx: Optional[int] = None) -> None:
        """Increment generation counter for an island"""
        idx = island_idx if island_idx is not None else self.current_island
        self.island_generations[idx] += 1
        logger.info(
            f"Island {idx} generation incremented to {self.island_generations[idx]}"
        )

    def should_migrate(self) -> bool:
        """Check if migration should occur based on generation counters"""
        max_generation = max(self.island_generations)
        return (
            max_generation - self.last_migration_generation
        ) >= self.migration_interval

    def migrate_programs(self) -> None:
        """
        Perform migration between islands

        This should be called periodically to share good solutions between islands
        """
        if len(self.islands) < 2:
            return

        logger.info("Performing migration between islands")

        for i, island in enumerate(self.islands):
            if len(island) == 0:
                continue

            # Select top programs from this island for migration
            island_programs = [
                self.programs[pid] for pid in island if pid in self.programs
            ]
            if not island_programs:
                continue

            # Sort by fitness (using combined_score or average metrics)
            island_programs.sort(
                key=lambda p: p.metrics['combined_score'],
                reverse=True,
            )

            # Select top programs for migration
            num_to_migrate = max(1, int(len(island_programs) * self.migration_rate))
            migrants = island_programs[:num_to_migrate]

            # Migrate to adjacent islands (ring topology)
            target_islands = [(i + 1) % len(self.islands), (i - 1) % len(self.islands)]

            for migrant in migrants:
                for target_island in target_islands:
                    # Create a copy for migration (to avoid removing from source)
                    migrant_copy = Program(
                        id=f"{migrant.id}_migrant_{target_island}",
                        code=migrant.code,
                        idea=migrant.idea,
                        language=migrant.language,
                        metrics=migrant.metrics.copy(),
                        parent_id=migrant.id,
                        iteration_found=migrant.iteration_found,
                        evolution_history=migrant.evolution_history,
                        report=migrant.report,
                        metadata={
                            **migrant.metadata,
                            "island": target_island,
                            "migrant": True,
                        },
                    )

                    # Add to target island
                    self.islands[target_island].add(migrant_copy.id)
                    self.programs[migrant_copy.id] = migrant_copy

                    logger.info(
                        f"Migrated program {migrant.id} from island {i} to island {target_island}"
                    )

        # Update last migration generation
        self.last_migration_generation = max(self.island_generations)
        logger.info(
            f"Migration completed at generation {self.last_migration_generation}"
        )

    def get_island_stats(self) -> List[dict]:
        """Get statistics for each island"""
        stats = []

        for i, island in enumerate(self.islands):
            island_programs = [
                self.programs[pid] for pid in island if pid in self.programs
            ]

            if island_programs:
                scores = [
                    p.metrics['combined_score']
                    for p in island_programs
                ]

                best_score = max(scores) if scores else 0.0
                avg_score = sum(scores) / len(scores) if scores else 0.0
                diversity = self._calculate_island_diversity(island_programs)
            else:
                best_score = avg_score = diversity = 0.0

            stats.append(
                {
                    "island": i,
                    "population_size": len(island_programs),
                    "best_score": best_score,
                    "average_score": avg_score,
                    "diversity": diversity,
                    "generation": self.island_generations[i],
                    "is_current": i == self.current_island,
                }
            )

        return stats

    def _calculate_island_diversity(self, programs: List[Program]) -> float:
        """Calculate diversity within an island"""
        if len(programs) < 2:
            return 0.0

        total_distance = 0
        comparisons = 0

        # Sample up to 10 programs for efficiency
        sample_size = min(10, len(programs))
        sample_programs = (
            random.sample(programs, sample_size)
            if len(programs) > sample_size
            else programs
        )

        for i, prog1 in enumerate(sample_programs):
            for prog2 in sample_programs[i + 1 :]:
                total_distance += Levenshtein.distance(prog1.code, prog2.code)
                comparisons += 1

        return total_distance / max(1, comparisons)

    def log_island_status(self) -> None:
        """Log current status of all islands"""
        stats = self.get_island_stats()
        logger.info("Island Status:")
        for stat in stats:
            current_marker = " *" if stat["is_current"] else "  "
            logger.info(
                f"{current_marker} Island {stat['island']}: {stat['population_size']} programs, "
                f"best={stat['best_score']:.4f}, avg={stat['average_score']:.4f}, "
                f"diversity={stat['diversity']:.2f}, gen={stat['generation']}"
            )