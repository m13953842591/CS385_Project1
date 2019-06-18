from .cnn import CNN
from .fisher import LDA
from .logistic import LogisticModel
from .svm import SVM
from .cnn import AlexNet


model_from_name = {'cnn': CNN(),
          'fisher': LDA(900),
          'svm_linear': SVM(kernel='linear'),
          'svm_rbf': SVM(kernel='rbf'),
          'svm_ploy': SVM(kernel='ploy'),
          'logistic': LogisticModel(),
          }


