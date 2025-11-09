### Quick Notes:

| Algorithm          | Type        | Speed       | Stability | Handles Large Data? | Common Use               |
| ------------------ | ----------- | ----------- | --------- | ------------------- | ------------------------ |
| Normal Eqn         | Closed-form | 🐢 Slow     | ❌ Poor    | ❌                   | Toy problems             |
| QR Decomp          | Numeric     | ⚡ Fast      | ✅ Good    | ⚙️ Medium           | SciPy/Sklearn internal   |
| SVD                | Numeric     | 🐢 Slowest  | ✅✅ Best   | ⚙️ Medium           | Stable exact solution    |
| Gradient Descent   | Iterative   | ⚡⚡ Scalable | ✅ Depends | ✅✅                  | Big data, deep learning  |
| Ridge / Lasso      | Regularized | ⚡           | ✅✅        | ⚙️                  | General ML               |
| Conjugate Gradient | Iterative   | ⚡           | ✅         | ✅✅                  | Sparse data, big systems |
