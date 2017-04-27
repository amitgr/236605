# 236605
This code requires the libraries:
  scikit-learn (and its prequisite)
  numpy
  scikit-neuralnetwork
  
  Modifications required:
    (*) scikit-learn: change if np_version < (1, 12, 0) -> if np_version != (1, 12, '0b1') and np_version < (1, 12, 0) in lib\site-packages\sklearn\utils\fixes.py
        due to a bug in the library. (See: https://github.com/scikit-learn/scikit-learn/issues/8034)
    (*) scikit-neuralnetwork: To work propelry with evaluation methods of scikit-learn, we changed lasagne/mlp.py:
      (-) cast(array, indices): is array is None -> if array is None or array.size == 0:
      (-) _iterate_data(self, batch_size, X, y=None, w=None, shuffle=False):  add line:
          `w = w if not w else numpy.asarray(w)` before `for index in range(0, total_size, batch_size):`
          
Running:
The runner generate the data for analysis (in spreadsheet format and dictionary).
There are 3 important runners:
  1. evaluator: for MPN-Brand-GTIN with SVM and Logistic Regression
  2. evaluator_mpn_brand: For MPN-Brand with Logistic Regression
  3. evaluator_with_nn: For MPN-Brand with Neural Network.
