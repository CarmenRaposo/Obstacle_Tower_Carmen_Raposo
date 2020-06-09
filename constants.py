from utils import otc_arg_parser

#Load parse parameters
parser = otc_arg_parser()
args = parser.parse_args()

if args.study:
    args.training_name = 'study'

# GLOBAL_PATH = args.results_dir + args.training_name + "/"
GLOBAL_PATH = args.results_dir