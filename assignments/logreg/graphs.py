import matplotlib.pyplot as plt

def populate_graph():

    """
    plt.figure()
    x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    train = [0.9652255639097744, 0.9614661654135338, 0.9558270676691729, 0.9511278195488722, 0.8984962406015038, 0.9088345864661654, 0.9351503759398496, 0.9464285714285714, 0.9483082706766918, 0.924812030075188]
    test = [0.9323308270676691, 0.8796992481203008, 0.8872180451127819, 0.8646616541353384, 0.849624060150376, 0.8571428571428571, 0.9172932330827067, 0.9022556390977443, 0.9022556390977443, 0.8721804511278195]
    plt.plot(x,train,'-r', label="Train Accuracy")
    plt.plot(x,test,'-b', label="Test Accuracy")
    plt.title("Steps vs Accuracy")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.close()
    """
    """
    plt.figure()
    x = [2,3,4,5,6,7,8,9,10,11,12]
    #train = [0.989662, 0.997180, 0.999060, 0.999060, 0.999060, 1.0, 1.0]
    #test = [0.939850, 0.954887, 0.939850, 0.947368, 0.947368, 0.932331, 0.939850]
    train = [0.986842,0.996241,0.997180,0.998120,0.999060,0.999060,0.999060,0.999060,0.999060,0.999060,0.999060]
    test = [0.932331,0.932331,0.947368,0.947368,0.947368,0.947368,0.947368,0.954887,0.954887,0.954887,0.954887]
    

    plt.plot(x,train,'-r', label="Train Accuracy")
    plt.plot(x,test,'-b', label="Test Accuracy")
    plt.title("TF-IDF Passes vs Accuracy")
    plt.xlabel("Passes")
    plt.ylabel("Accuracy")
    plt.axis([2,12,0.93,1.02])
    plt.legend(loc="center right")
    plt.show()
    plt.close()
    """
    plt.figure()
    x=[1,2,3,4,5,6,7,8]
    train =[0.965226,0.985902,0.989662,0.990602,0.995301,0.997241,1.0,1.0]
    test = [0.932331,0.954887,0.954887,0.954887,0.962406,0.962406,0.962406,0.962406]
    plt.plot(x,train,'-r', label="Train Accuracy")
    plt.plot(x,test,'-b', label="Test Accuracy")
    plt.title("Passes vs Accuracy")
    plt.xlabel("Passes")
    plt.ylabel("Accuracy")
    plt.axis([1,8,0.93,1.02])
    plt.legend()
    plt.show()
    plt.close()
    
populate_graph()
