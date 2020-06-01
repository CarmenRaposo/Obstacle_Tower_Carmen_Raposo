from utils import otc_arg_parser

#Load parse parameters
parser = otc_arg_parser()
args = parser.parse_args()

GLOBAL_PATH = args.results_dir + "_" + args.training_name + "/"