from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys
import tensorflow
import numpy as np

from extract_training_data import FeatureExtractor, State, apply_sequence

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = tensorflow.keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            features = self.extractor.get_input_representation(words, pos, state)
            features = features.reshape(1,-1)
            transition_scores = self.model.predict_on_batch(features)
            found_transition = False
            i = 1
            sorted = np.argsort(transition_scores)
            while not found_transition:
                found_transition = True
                index_highest = sorted[-1][-i]
                transition = self.output_labels[index_highest]
                operation = transition[0]
                label = transition[1]

                #arc-left/arc-right not allow if stack is empty
                if 'arc' in operation and len(state.stack) == 0:
                    found_transition = False
                #shifting the only word out of buffer is also illegal, unless the stack is empty
                elif 'shift' in operation  and len(state.buffer) == 1 and len(state.stack) !=0:
                    found_transition = False
                #root node cannot be target of left arc 
                elif 'left_arc' in operation and transition[1] == 'root':
                    found_transition = False

                i = i+1
            
            #perform operation:
            if 'left_arc' in operation:
                state.left_arc(label)
            elif 'shift' in operation:
                state.shift()
            elif 'right_arc' in operation: 
                state.right_arc(label)
            else:
                print("this is issue!!!!")

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
