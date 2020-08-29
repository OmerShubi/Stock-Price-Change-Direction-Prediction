# Stock Price Change Direction Prediction 

This code implements and demonstrates the use of different models 
and algorithms to predict the direction of change of stock price.

As we describe in the paper, part 1 compares unstructured to structured models, 
where the structure comes into play by grouping the days into weeks.
Part 2 implements structured prediction as well, this time by looking 
at the connections between different companies on the same day.

The classifiers include

- Perceptron
- Structured (Chain CRF) Perceptron
- LSTM Neural Network
- MRF + Belief Propagation

We implement several aspects and build upon basic skeletons from sklearn, pystruct and pytorch for 
the remaining algorithms and training and inference stages.
Our work includes the preprocessing stages - for each day, for each week, 
for each company, and for each company pair,
the LSTM Neural Network training phase, structuring and creating the MRF, 
adapting the BP for multi-day inference, result analysis and more.

 
To reproduce the results:

1. Clone the code
2. Create a suitable environment `conda create --file env.yaml`
3. Activate the env `conda activate ML36`
4. Run the main script `python main.py`

Notes:
- Hyperparameters are defined in `utils/Params.py` and can be freely changed.
- Full run log `log_file.log` is automatically created and filled.
- To use different companies simply add the data to the `data` folder, following the naming convention `<stock_name>.us.txt`
and update the `STOCK_NAMES` parameter in the `Params.py` config file.
