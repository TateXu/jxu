
import argparse
import pdb
import argparse

# https://docs.python.org/3/library/argparse.html#module-argparse

parser = argparse.ArgumentParser(description='Processing NIBS data.', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-lr', '--load_raw', action='store_true', help='Property: flag;\nDeafult=False;\nFunc: Loading completely raw data;')
parser.add_argument('-sc', '--save_clean', action='store_true', help='Property: flag;\nDeafult=False;\nFunc: Saving filtered& cropped raw data;')
parser.add_argument('-lc', '--load_clean', action='store_true', help='Property: flag;\nDefault=False;\nFunc: Loading filtered& cropped raw data;')
parser.add_argument('-ec', '--epoch_clean', dest='bar', action='store_true', help='Property: flag;\nDeafult=False;\nFunc: Epoching filtered& cropped raw data;')
parser.add_argument("-f", "--file", dest="filename",
                    help="write report to FILE", metavar="FILE")
parser.add_argument("-q", "--quiet",
                    action="store_false", dest="verbose", default=True,
                    help="don't print status messages to stdout")
parser.add_argument("--name_of_variable", help="echo the string you use here", type=str, choices=['a', 'b'])

parser.add_argument("--name_of_variable_1", help="echo the string you use here", type=int, choices=[1, 2, 3])

args = parser.parse_args()  # '--load_raw --save_clean --load_clean --epoch_clean'.split()

locals().update(vars(args))

print(args.name_of_variable)
print(args.name_of_variable == None)
print(args.bar)

pdb.set_trace()