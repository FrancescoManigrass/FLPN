from logltn.core import Variable, Predicate, Constant, Function, Connective, diag, undiag, Quantifier, \
    LTNObject, process_ltn_objects, LambdaModel
import torch
import logltn.fuzzy_ops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")