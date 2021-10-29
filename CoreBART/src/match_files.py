
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", default=None, type=str, required=True,
                            help="Model type: e.g. roberta")
    parser.add_argument("--file2", default=None, type=str, required=True,
                            help="Path to pre-trained model: e.g. roberta-base")
    args = parser.parse_args()
    with open(args.file1) as f1 , open(args.file2) as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        assert(len(lines1)==len(lines2))
        cnt = 0
        for idx in range(len(lines1)):
            if lines1[idx].strip()==lines2[idx].strip():
                cnt+=1
    print("count = ", str(cnt), "total = ", str(len(lines1)))
    print("accuracy = ", str(round(cnt/len(lines1) * 100, 4)))

if __name__ == "__main__":
    main()
    
