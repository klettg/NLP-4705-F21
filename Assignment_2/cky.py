"""
COMS W4705 - Natural Language Processing
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar
        self.table = defaultdict()
        self.probs = defaultdict()
        self.basicTable = defaultdict()

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """

        #initialize empty dictionary values for table
        sent_length = len(tokens)
        for i in range(0, sent_length + 1):
            for j in range(0, sent_length + 1 ):
                self.basicTable[(i, j)] = []

        #initialization with words:
        for i in range(0, len(tokens)):
            token = tokens[i]
            dict_key = tuple(token.strip().split())
            symbol_values = self.grammar.rhs_to_rules[dict_key]
            just_symbols = [x[0] for x in symbol_values]
            self.basicTable[(i, i + 1)] = just_symbols

        for length in range(2, len(tokens) + 1):
            for i in range(0, (len(tokens)-length) + 1):
                j = i + length
                for k in range(i+1, j):
                    left_arrow_symbols = self.basicTable[(i, k)]
                    up_arrow_symbols = self.basicTable[(k, j)]
                    for lhs_symbol in left_arrow_symbols:
                        for up_arrow_symbol in up_arrow_symbols:
                            potential_rule = (lhs_symbol, up_arrow_symbol)
                            if potential_rule in self.grammar.rhs_to_rules:
                                rules = self.grammar.rhs_to_rules[potential_rule]
                                rule_names = [x[0] for x in rules]
                                curr_values = self.basicTable[(i, j)]
                                added_together = rule_names + curr_values
                                self.basicTable[(i, j)] = added_together

        if self.grammar.startsymbol in self.basicTable[(0, len(tokens))]:
            return True

        return False 
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = defaultdict()
        probs = defaultdict()

        # initialize empty dictionary values for table
        sent_length = len(tokens)
        for i in range(0, sent_length + 1):
            for j in range(0, sent_length + 1):
                table[(i, j)] = defaultdict()
                probs[(i, j)] = defaultdict()

        # base case
        for i in range(0, len(tokens)):
            token = tokens[i]
            dict_key = tuple(token.strip().split())
            symbol_values = self.grammar.rhs_to_rules[dict_key]
            span_dictionary = table[(i, i+1)]
            prob_dictionary = probs[(i, i+1)]
            for symbol_value in symbol_values:
                log_prob = math.log(symbol_value[2])
                symbol = symbol_value[0]
                terminal_string = ''.join(symbol_value[1])
                prob_dictionary[symbol] = log_prob
                span_dictionary[symbol] = terminal_string

            # reset table
            table[(i, i + 1)] = span_dictionary
            probs[(i, i + 1)] = prob_dictionary

        for length in range(2, len(tokens) + 1):

            for i in range(0, (len(tokens)-length) + 1):
                j = i + length
                for k in range(i+1, j):
                    span_dictionary_l_arrow = table[(i, k)]
                    prob_dictionary_l_arrow = probs[(i, k)]
                    span_dictionary_r_arrow = table[(k, j)]
                    prob_dictionary_r_arrow = probs[(k, j)]

                    for symbol_key_r_arrow in span_dictionary_r_arrow:

                        for symbol_key_l_arrow in span_dictionary_l_arrow:

                            potential_rule = (symbol_key_l_arrow, symbol_key_r_arrow)

                            if potential_rule in self.grammar.rhs_to_rules:

                                # now we add to table with back pointers
                                rules = self.grammar.rhs_to_rules[potential_rule]
                                for rule in rules:
                                    prob_left = prob_dictionary_l_arrow[symbol_key_l_arrow]
                                    prob_right = prob_dictionary_r_arrow[symbol_key_r_arrow]
                                    curr_transition_prob = math.log(rule[2])
                                    entry_prob = curr_transition_prob + prob_right + prob_left

                                    rule_name = rule[0]
                                    # check if there already exists a prob for this type of rule at this span
                                    if rule_name in table[(i, j)]:
                                        # compare probabilities
                                        existing_prob = probs[(i, j)][rule_name]
                                        # if this split is higher prob, include this instead
                                        if entry_prob > existing_prob:
                                            probs[(i, j)][rule_name] = entry_prob

                                            # set back pointers to left, right
                                            table[(i, j)][rule_name] = ((symbol_key_l_arrow, i, k), (symbol_key_r_arrow, k, j))
                                    else:
                                        table[(i, j)][rule_name] = ((symbol_key_l_arrow, i, k), (symbol_key_r_arrow, k, j))
                                        probs[(i, j)][rule_name] = entry_prob
        return table, probs


def get_tree(chart, i,j,nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    final_output = get_tree_helper(chart, nt, i, j)
    return final_output

#def get_tree_helper2(chart, root, i, j, output):
 #   updated = (output, left, right)


def get_tree_helper(chart, root, i, j):
    #output += '(\''
    #output += root
    #output += '\', '
    if i == j or type(chart[(i, j)][root]) == str:
        #output += '\''
        #output += chart[(i,j)][root]
        #output += '\''
        #output += ')'
        temp = chart[(i, j)][root]
        return (root, temp)

    value = chart[(i, j)][root]
    left = get_tree_helper(chart, value[0][0], value[0][1], value[0][2])
    #output += ', '
    right = get_tree_helper(chart, value[1][0], value[1][1], value[1][2])
    #output += ')'
    return (root, left, right)


       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        print(parser.is_in_language(toks))
        table, probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        get_tree(table, 0, len(toks), grammar.startsymbol)
        
