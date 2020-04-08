# 1. Positional arguments

# string 
parser.add_argument("name_of_variable", help="echo the string you use here")

# number 
parser.add_argument("name_of_variable", help="echo the string you use here", type=int)



# 2. Optional arguments
# Have to specify a string, otherwise None
parser.add_argument("--name_of_variable", help="echo the string you use here")

# Have to specify a number, otherwise None
parser.add_argument("--name_of_variable", help="echo the string you use here", type=int)


# A bool flag, default value = ~action, i.e., without specifying a value. Otherwise, == action
# Cannot specifying a valuem but only show the option, i.e, python xxx.py --name_of_variable
parser.add_argument("--name_of_variable", help="echo the string you use here", action="store_true")


# Short options: 
parser.add_argument("-nov", "--name_of_variable", help="echo the string you use here", action="store_true")

# Have to specify a number or string within a given list of options, otherwise None
parser.add_argument("--name_of_variable", help="echo the string you use here", type=int, choices=[1, 2, 3])

parser.add_argument("--name_of_variable", help="echo the string you use here", type=str, choices=['a', 'b'])











