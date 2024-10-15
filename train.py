# import libraries
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split

def main(args):
	# TO DO: enable autologging


    # read data
    df = get_data(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    # evaluate model
    eval_model(model, X_test, y_test)

# function that reads the data
def get_data(path):
    print("Reading data...")
    ratings_all = pd.read_csv(path)
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(ratings_all[['userId', 'movieId', 'rating']], reader)
    
    return data

# function that splits the data
def split_data(data):
    print("Splitting data...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    return trainset, testset

# function that trains the model
def train_model(trainset):
    print("Training model...")
    model = SVD(n_factors= 100, n_epochs= 50, lr_all= 0.01, reg_all= 0.02, random_state= 42).fit(trainset)

    return model

# function that evaluates the model
def eval_model(model, testset):
    # calculate accuracy
    predictions = model.test(testset)
    accuracy.rmse(predictions)
    print('rmse:', accuracy)

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.02)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
