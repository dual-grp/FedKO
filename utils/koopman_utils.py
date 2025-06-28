from scipy.linalg import hankel
import numpy as np

def create_time_delay_embedding(data, m):
    """ Create time-delay embedded matrices for DMD. """
    N = len(data)
    X = np.array([data[i:i+m-1] for i in range(N - m)])
    Y = np.array([data[i+1:i+m] for i in range(N - m)])
    return X.T, Y.T

# Apply DMD on X and Y
def dmd(X, Y, r=None):
    """
    Compute the Dynamic Mode Decomposition of two matrices X and Y.

    Parameters:
    - X: Matrix at time t
    - Y: Matrix at time t+1
    - r: Rank for truncation

    Returns:
    - Atilde: Reduced A matrix (Koopman operator in reduced space)
    - U: Left singular vectors from SVD of X
    - Lambda: Eigenvalues of Atilde
    - W: Eigenvectors of Atilde
    """
    U, S_vals, Vh = np.linalg.svd(X, full_matrices=False)
    if r is not None:
        U = U[:, :r]
        S = np.diag(S_vals[:r])
        V = Vh[:r, :].conj().T  # Note the conjugate transpose to get V
    else:
        S = np.diag(S_vals)
        V = Vh.conj().T  # Note the conjugate transpose to get V
    # Compute the approximation of the Koopman operator
    """
    Atilde = np.dot(U.conj().T,
                    np.dot(Y, 
                            np.dot(Vh,
                                    np.linalg.inv(np.diag(S))
                                    )
                            )
                    )

    print("U shape:", U.shape)
    print("Y shape:", Y.shape)
    print("V shape:", V.shape)
    print("S shape:", S.shape)
    """
    #Atilde = U.conj().T.dot(Y).dot(V).dot(np.linalg.inv(S))
    Atilde = U.conj().T.dot(Y).dot(V).dot(np.linalg.inv(S))
    #print(type(Atilde))
    Lambda, W = np.linalg.eig(Atilde)
    return Atilde, U, Lambda, W

def predict_next_state(Atilde, U, W, Lambda, latest_snapshot, t=1):
    """
    Predict the next state based on the DMD.
    
    Parameters:
    - Atilde: Reduced A matrix (Koopman operator in reduced space)
    - U: Left singular vectors from SVD
    - W: Eigenvectors of Atilde
    - Lambda: Eigenvalues of Atilde
    - latest_snapshot: Last known state
    - t: Time steps ahead for prediction
    
    Returns:
    - predicted_snapshot: Predicted state t time steps ahead
    """
    # Step 2: Initial condition in reduced space
    b = np.linalg.inv(W).dot(U.conj().T).dot(latest_snapshot)
    
    # Step 3: Time evolution
    b_t = np.power(Lambda, t) * b
    
    # Step 4: Convert back to original space
    predicted_snapshot = U.dot(W).dot(b_t)
    
    return predicted_snapshot

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def step(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        self.prev_error = error
        return output
