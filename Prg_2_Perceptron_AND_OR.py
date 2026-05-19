import numpy as np
def step_function(x):
    return 1 if x>=0 else 0
class Perceptron:
    def __init__(self,input_size,learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias=0
        self.learning_rate=learning_rate
    def predict(self,inputs):
        summation=np.dot(inputs,self.weights)+self.bias
        return step_function(summation)
    def train(self,X,y,epochs=10):
        for epoch in range(epochs):
            for i in range(len(X)):
                 prediction=self.predict(X[i])
                 error=y[i]- prediction
                 self.weights+=self.learning_rate*error*X[i]
                 self.bias+=self.learning_rate*error
            print(f"Epoch {epoch+1}: Weights :{self.weights}, Bias: {self.bias}")
        print(f"\nFinal Weights after training :{self.weights}")
        print(f"\nFinal bias after training :{self.bias}")

X_and=np.array([[0,0],[0,1],[1,0],[1,1]])
y_and=np.array([0,0,0,1])
X_or=np.array([[0,0],[0,1],[1,0],[1,1]])
y_or=np.array([0,1,1,1])

perceptron_and=Perceptron(input_size=2)
print("Training perceptron for AND gate")
perceptron_and.train(X_and,y_and)

perceptron_or=Perceptron(input_size=2)
print("Training perceptron for OR gate")
perceptron_or.train(X_or,y_or)

print("\n Testing predictions")
test_inputs=np.array([[0,0],[0,1],[1,0],[1,1]])

print("\n Predictions for AND gate")
for test in test_inputs:
    print(f"Input : {test}-> Prediction:{perceptron_and.predict(test)}")

print("\n Predictions for OR gate")
for test in test_inputs:
    print(f"Input : {test}-> Prediction:{perceptron_or.predict(test)}")
