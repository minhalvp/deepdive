import argparse

def main():
    parser = argparse.ArgumentParser(description='DeepDive command line interface.')
    parser.add_argument('deploy', type=str, help='Deploy model from folderpath.')
    args = parser.parse_args()

if __name__ == '__main__':
    main()
