import sys
import random
from rec2vec import graph
from gensim.models import Word2Vec
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import time
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

def process(args):
    # Create a graph from the training set
    nodedict = graph.records_to_graph()

    # Build the model using DeepWalk and Word2Vec
    G = graph.load_adjacencylist("out.adj", undirected=True)

    ####################################################################################################################################################################
    # Code Written for BI Project : Author : Himangshu Ranjan Borah(hborah)
    ####################################################################################################################################################################

    # call the build_deepwalk_corpus function
    # Take and populate the arguments from the command lines.                     
    generated_walks = graph.build_deepwalk_corpus(G = G, num_paths = args.number_walks, path_length = args.walk_length, alpha=0, rand=random.Random(0))
    # Call word2vec to build the model.
    # print generated_walks
    # The structure Looks like ['32173', '32168'], ['124010', '22676'], ['17792', '72925'],
    model = Word2Vec(generated_walks, size=args.representation_size, window=args.window_size, min_count=0, workers=args.workers)
    

    ####################################################################################################################################################################
    # Code Written for BI Project : Author : Himangshu Ranjan Borah(hborah)
    ####################################################################################################################################################################

    # Perform some evaluation of the model on the test dataset
    with open("./data/test_user_ratings.dat") as fin:
        fin.next()
        groundtruth = [line.strip().split("\t")[:3] for line in fin]    # (user, movie, rating)
    tr = [int(round(float(g[2]))) for g in groundtruth]
    pr = [predict_rating(model, nodedict, "u"+g[0], "m"+g[1]) for g in groundtruth]

    print "MSE = %f" % mean_squared_error(tr, pr)
    print "accuracy = %f" % accuracy_score(tr, pr)
    cm = confusion_matrix(tr, pr, labels=range(1,6))
    print cm


def predict_rating(model, nodedict, user, movie):
    """
    Predicts the rating between a user and a movie by finding the movie-rating node with the highest
    similarity to the given user node.
    Loops through the five possible movie-rating nodes and finds the node with the highest similarity to the user.
    
    Returns an integer rating 1-5.
    """
    ####################################################################################################################################################################
    # Code Written for BI Project : Author : Himangshu Ranjan Borah(hborah)
    ####################################################################################################################################################################

    # Get the actual Node ID
    user_id = nodedict[user].id
    potential_rating_node_ids = list()
    # Pull out the 5 rating nodes.
    for i in xrange(1,6):
        potential_rating_node_ids.append(nodedict[movie + '_' + str(i)].id)

    #print user_id, potential_rating_node_ids   
    max_sim = float("-inf")
    max_rating = 0   
    for i, rating_node_id in enumerate(potential_rating_node_ids):
        #print i, rating_node_id, user_id
        # String of the ids or just the ids? Used String for building the model, so use string here! Uniformity.
        # However, this doen't matter, just we need to be consistent.
        curr_sim = model.similarity(str(rating_node_id), str(user_id))
        #print curr_sim
        if curr_sim > max_sim:
            max_sim = curr_sim
            max_rating = i + 1

    #print max_rating, max_sim
    #exit(0)        
    
    return max_rating    

    ####################################################################################################################################################################
    # Code Written for BI Project : Author : Himangshu Ranjan Borah(hborah)
    ####################################################################################################################################################################




def main():
    parser = ArgumentParser("rec2vec", 
        formatter_class=ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('--number-walks', default=10, type=int,
        help='Number of random walks to start at each node')
    parser.add_argument('--walk-length', default=40, type=int,
        help='Length of the random walk started at each node')
    parser.add_argument('--seed', default=0, type=int,
        help='Seed for random walk generator.')   
    parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
        help='Size to start dumping walks to disk, instead of keeping them in memory.')
    parser.add_argument('--window-size', default=5, type=int,
        help='Window size of skipgram model.')
    parser.add_argument('--workers', default=1, type=int,
        help='Number of parallel processes.')        
    parser.add_argument('--representation-size', default=64, type=int,
        help='Number of latent dimensions to learn for each node.')
    
    parser.set_defaults(csv_to_graph=True, loo=True)
    args = parser.parse_args()
    
    process(args)
    

if __name__=="__main__":
    sys.exit(main())
