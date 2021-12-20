"""
COMS W4705 - Natural Language Processing
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """

        for key in self.lhs_to_rules:
            list = self.lhs_to_rules[key]
            sum = math.fsum(x[2] for x in list)
            isclose = math.isclose(1, sum, rel_tol = 1e-09, abs_tol = 0.0)
            if not isclose:
                return False

            # if LHS key is not upper case, this is incorrect format
            # check if RHS has correct format too
            for rule in list:
                lhs_name = rule[0]
                if not lhs_name.isupper():
                    return False

                rhs_tuple = rule[1]

                # if terminal, rhs has only one element, and is lower case
                if len(rhs_tuple) == 1:
                    if rhs_tuple[0].isupper():
                        return False

                # if non-terminal, should have length 2 and both be upper case
                if len(rhs_tuple) == 2:
                    if not rhs_tuple[0].isupper() and not rhs_tuple[1].isupper():
                        return False

                if len(rhs_tuple) !=2 and len(rhs_tuple) != 1:
                    return False

        return True


if __name__ == "__main__":
    # with open('atis3.pcfg', 'r') as grammar_file:
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        output = grammar.verify_grammar()
        if output:
            print('verified valid grammar')
        else:
            print('Error, invalid grammar!')
