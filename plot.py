import matplotlib.pyplot as plt

def graph(X, Y):
  plt.plot(X, Y, label = "Predicted radon values")
  
  
  plt.xlim([0, 100])
  plt.ylim([0, 9000])
  plt.figure(figsize = (30, 12))
  plt.show()
  
