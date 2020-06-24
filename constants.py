from utils import otc_arg_parser

#Load parse parameters
parser = otc_arg_parser()
args = parser.parse_args()

if args.study:
    args.training_name = 'study'

# GLOBAL_PATH = args.results_dir + args.training_name + "/"
GLOBAL_PATH = args.results_dir

OBSTACLE_TOWER_PATH = '/home/home/Data/Carmen/py_workspace/ObstacleTower_v3/ObstacleTower-v3.1/obstacletower.x86_64'

IMAGE_SIZE = 84

#Labels for State Classifier
LABELS = ('closed door', 'locked door', 'boxed door', 'open door', 'key', 'box', 'hurtle', 'orb',
          'goal', 'box target', 'box undo')

NUM_LABELS = len(LABELS)