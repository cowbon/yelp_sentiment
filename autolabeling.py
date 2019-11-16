import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input training file', default=os.path.join(workdir, 'pratice.json'))
    args = parser.parse_args()
    
if __name__ == '__main__':
    main()
