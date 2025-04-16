import scipy.io

def main():
    data = scipy.io.loadmat('tennis_data.mat')
    return data['G']

if __name__ == "__main__":
    G = main()
    print(G - 1)
    
