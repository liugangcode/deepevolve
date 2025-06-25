python deepevolve.py \
    query="'You are an expert mathematician. Your task is to improve an algorithm that maximizes the sum of circle radii in the circle-packing problem within a unit square, using between 26 and 32 circles. Do not develop neural-network-based models. The algorithm must produce exact, valid packings that satisfy these constraints: circles not overlap and must remain entirely within the square.'" \
    problem="circle_packing" \
    checkpoint="ckpt" \
    checkpoint_interval=20

python deepevolve.py \
    query="Your task is to improve the graph rationalization method for more accurate and interpretable molecular property prediction" \
    problem="molecule" \
    max_iterations=100

python deepevolve.py \
    query="'Your task is to improve the nucleus detection models in a Kaggle competition within a compute budget of an A6k GPU with a maximum runtime of 30 minutes. You should significantly improve both the performance of the initial idea and its efficiency.'" \
    problem="nuclei_image"


python deepevolve.py \
    query="Your task is to improve the performance of the winning solution for the Kaggle competition on Parkinson disease progression prediction. You may propose a completely new approach that differs from the winning solution if you believe it will perform better." \
    problem="parkinson_disease"

python deepevolve.py \
    query="'Your task is to significantly improve polymer property prediction for five properties in the competition. The input SMILES strings are the monomer structures of polymers, using asterisks (*) to mark the polymerization points. You should improve the initial idea by focusing on how to better incorporate polymerization inductive bias into the models to improve the weighted mean absolute error and the R-squared value for each property. You should explore different ways to exploit polymer structures or properties and find the best. Your time budget is 30 minutes.  Make sure you implement your idea within the time limit rather than create a placeholder.'" \
    problem="polymer"

python deepevolve.py \
    query="'Your task is to fine-tune Patent BERT to predict semantic similarity between phrase pairs from U.S. patents. Improve model performance, optimize training time and inference latency, and ensure the fixed three-epoch run finishes in thirty minutes. Focus solely on technical model and algorithm development. No legal-style assistance in your response.'" \
    problem="usp_p2p"
