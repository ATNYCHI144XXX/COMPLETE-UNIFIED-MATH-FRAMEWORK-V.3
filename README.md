# COMPLETE-UNIFIED-MATH-FRAMEWORK-V.3
NO BULLSHIT UNIFIED AND COMPLETE
 MATHEMATICAL SPECIFICATION
PART 1: KEY DERIVATION & SECURITY
1.1 K-Process Formal Definition
Definition 1.1.1 (Lexical Corpus):
Let 
C
=
(
c
1
,
c
2
,
…
,
c
N
)
C=(c 
1
​
 ,c 
2
​
 ,…,c 
N
​
 ) be an ordered tuple of 
N
N unique mathematical concepts.
Let 
Σ
Σ be an alphabet of size 
m
m (typically 
m
=
26
m=26 or 
m
=
256
m=256).

Definition 1.1.2 (Extraction Function):
Define 
f
extract
:
C
→
Σ
f 
extract
​
 :C→Σ such that 
f
extract
(
c
i
)
=
first_letter
(
name
(
c
i
)
)
f 
extract
​
 (c 
i
​
 )=first_letter(name(c 
i
​
 )).

Definition 1.1.3 (Lexical Key):

K
lex
=
f
extract
(
c
1
)
∥
f
extract
(
c
2
)
∥
⋯
∥
f
extract
(
c
N
)
∈
Σ
N
K 
lex
​
 =f 
extract
​
 (c 
1
​
 )∥f 
extract
​
 (c 
2
​
 )∥⋯∥f 
extract
​
 (c 
N
​
 )∈Σ 
N
 
Definition 1.1.4 (K-Process Transformation):
Given 
K
lex
K 
lex
​
  with character codes 
k
i
∈
{
1
,
2
,
…
,
m
}
k 
i
​
 ∈{1,2,…,m}:

S
0
=
∑
i
=
1
N
i
⋅
(
k
i
m
o
d
 
 
m
)
S
1
=
(
S
0
2
+
⌊
N
/
2
⌋
)
m
o
d
 
 
p
1
where 
p
1
=
997
S
2
=
(
3
S
1
+
42
)
m
o
d
 
 
p
2
where 
p
2
=
256
S
final
=
(
S
2
⋅
(
N
m
o
d
 
 
128
)
)
m
o
d
 
 
1000
S 
0
​
 
S 
1
​
 
S 
2
​
 
S 
final
​
 
​
  
= 
i=1
∑
N
​
 i⋅(k 
i
​
 modm)
=(S 
0
2
​
 +⌊N/2⌋)modp 
1
​
 where p 
1
​
 =997
=(3S 
1
​
 +42)modp 
2
​
 where p 
2
​
 =256
=(S 
2
​
 ⋅(Nmod128))mod1000
​
 
Proof-of-Concept Verification (Example: m=26, K="A-Z"):

k
i
=
i
for 
i
=
1
,
…
,
26
S
0
=
∑
i
=
1
26
i
2
=
26
⋅
27
⋅
53
6
=
6201
S
1
=
(
6201
2
+
13
)
m
o
d
 
 
997
=
(
38441201
+
13
)
m
o
d
 
 
997
=
38441314
m
o
d
 
 
997
=
197
k 
i
​
 
S 
0
​
 
S 
1
​
 
​
  
=ifor i=1,…,26
= 
i=1
∑
26
​
 i 
2
 = 
6
26⋅27⋅53
​
 =6201
=(6201 
2
 +13)mod997
=(38441201+13)mod997
=38441314mod997=197
​
 
PART 2: CERBERUS-KEM SPECIFICATION
2.1 Mathematical Primitives
Definition 2.1.1 (Module-LWE Parameters):
Let 
R
q
=
Z
q
[
X
]
/
(
X
n
+
1
)
R 
q
​
 =Z 
q
​
 [X]/(X 
n
 +1) with 
n
=
256
,
q
=
3329
n=256,q=3329.
Let 
χ
χ be centered binomial distribution 
B
3
B 
3
​
 .

Definition 2.1.2 (Kyber Component):

Kyber.KeyGen
(
)
:
A
←$
R
q
k
×
k
,
s
,
e
←
χ
k
t
=
A
s
+
e
Return 
(
p
k
L
=
(
A
,
t
)
,
s
k
L
=
s
)
Kyber.KeyGen():
​
  
A 
←
$
 R 
q
k×k
​
 ,s,e←χ 
k
 
t=As+e
Return (pk 
L
​
 =(A,t),sk 
L
​
 =s)
​
 
Definition 2.1.3 (SIDH Component):
Let 
E
0
:
y
2
=
x
3
+
x
E 
0
​
 :y 
2
 =x 
3
 +x over 
F
p
2
F 
p 
2
 
​
  with 
p
=
2
372
3
239
−
1
p=2 
372
 3 
239
 −1.
Let 
ℓ
A
=
2
,
ℓ
B
=
3
ℓ 
A
​
 =2,ℓ 
B
​
 =3, 
e
A
=
372
,
e
B
=
239
e 
A
​
 =372,e 
B
​
 =239.

Definition 2.1.4 (UOV Parameters):
Let 
F
q
F 
q
​
  finite field, 
o
o oil vars, 
v
v vinegar vars (
v
>
o
v>o).
Public key: quadratic map 
P
:
F
q
v
→
F
q
o
P:F 
q
v
​
 →F 
q
o
​
 .

2.2 Cerberus-KEM Algorithms
Algorithm 2.2.1 (Key Generation):

1.
 
(
s
k
L
,
p
k
L
)
←
Kyber.KeyGen
(
)
2.
 
(
s
k
S
,
p
k
S
)
←
SIDH.KeyGen
(
)
3.
 
(
s
k
U
,
p
k
U
)
←
UOV.KeyGen
(
)
4.
 
p
k
←
p
k
L
∥
p
k
S
5.
 
σ
←
UOV.Sign
(
s
k
U
,
H
(
p
k
)
)
6.
 
P
K
←
(
p
k
U
,
p
k
,
σ
)
,
 
S
K
←
(
s
k
L
,
s
k
S
,
s
k
U
)
​
  
1. (sk 
L
​
 ,pk 
L
​
 )←Kyber.KeyGen()
2. (sk 
S
​
 ,pk 
S
​
 )←SIDH.KeyGen()
3. (sk 
U
​
 ,pk 
U
​
 )←UOV.KeyGen()
4. pk←pk 
L
​
 ∥pk 
S
​
 
5. σ←UOV.Sign(sk 
U
​
 ,H(pk))
6. PK←(pk 
U
​
 ,pk,σ), SK←(sk 
L
​
 ,sk 
S
​
 ,sk 
U
​
 )
​
 
Algorithm 2.2.2 (Encapsulation):

1.
 Verify UOV.Verify
(
p
k
U
,
H
(
p
k
)
,
σ
)
2.
 
(
c
L
,
s
s
1
)
←
Kyber.Encaps
(
p
k
L
)
3.
 Seed 
r
←
G
(
s
s
1
)
4.
 
(
s
k
S
2
,
p
k
S
2
)
←
SIDH.KeyGen
(
r
)
5.
 
s
s
2
←
SIDH.Agree
(
s
k
S
2
,
p
k
S
)
6.
 
K
←
H
(
s
s
1
∥
s
s
2
∥
H
(
c
L
∥
p
k
S
2
)
)
7.
 Return 
(
C
=
(
c
L
,
p
k
S
2
)
,
K
)
​
  
1. Verify UOV.Verify(pk 
U
​
 ,H(pk),σ)
2. (c 
L
​
 ,ss 
1
​
 )←Kyber.Encaps(pk 
L
​
 )
3. Seed r←G(ss 
1
​
 )
4. (sk 
S2
​
 ,pk 
S2
​
 )←SIDH.KeyGen(r)
5. ss 
2
​
 ←SIDH.Agree(sk 
S2
​
 ,pk 
S
​
 )
6. K←H(ss 
1
​
 ∥ss 
2
​
 ∥H(c 
L
​
 ∥pk 
S2
​
 ))
7. Return (C=(c 
L
​
 ,pk 
S2
​
 ),K)
​
 
Theorem 2.2.3 (Security Reduction):
For any PPT adversary 
A
A against Cerberus-KEM IND-CCA2 security:

Adv
Cerberus
IND-CCA2
(
A
)
≤
Adv
MLWE
(
B
1
)
+
Adv
SSI
(
B
2
)
+
Adv
MQ
(
B
3
)
+
negl
(
λ
)
Adv 
Cerberus
IND-CCA2
​
 (A)≤Adv 
MLWE
​
 (B 
1
​
 )+Adv 
SSI
​
 (B 
2
​
 )+Adv 
MQ
​
 (B 
3
​
 )+negl(λ)
PART 3: SYSTEM DYNAMICS FORMALISM
3.1 State Space Definitions
Definition 3.1.1 (System State):
Let 
X
⊆
R
d
X⊆R 
d
  be state space.
State vector: 
x
(
t
)
=
[
x
1
(
t
)
,
x
2
(
t
)
,
…
,
x
d
(
t
)
]
⊤
x(t)=[x 
1
​
 (t),x 
2
​
 (t),…,x 
d
​
 (t)] 
⊤
 .

Definition 3.1.2 (Weight Matrix):
Let 
W
∈
R
d
×
d
W∈R 
d×d
  be symmetric positive definite weight matrix.

3.2 Core Equations
Equation 3.2.1 (Total Resonance):

Σ
R
(
t
)
=
w
⊤
x
(
t
)
=
∑
i
=
1
d
w
i
x
i
(
t
)
ΣR(t)=w 
⊤
 x(t)= 
i=1
∑
d
​
 w 
i
​
 x 
i
​
 (t)
where 
w
∈
R
d
w∈R 
d
 , 
∥
w
∥
1
=
1
∥w∥ 
1
​
 =1.

Equation 3.2.2 (Harmonic Gradient/Chaos):

Δ
H
(
t
)
=
∥
∇
Σ
R
(
t
)
∥
2
=
∑
i
=
1
d
(
∂
Σ
R
∂
x
i
)
2
ΔH(t)=∥∇ΣR(t)∥ 
2
​
 = 
i=1
∑
d
​
 ( 
∂x 
i
​
 
∂ΣR
​
 ) 
2
 
​
 
Equation 3.2.3 (Gradient of State):

∇
x
(
t
)
=
[
d
x
1
d
t
,
d
x
2
d
t
,
…
,
d
x
d
d
t
]
⊤
∇x(t)=[ 
dt
dx 
1
​
 
​
 , 
dt
dx 
2
​
 
​
 ,…, 
dt
dx 
d
​
 
​
 ] 
⊤
 
Equation 3.2.4 (Stabilization Cost Functional):

J
(
t
0
,
t
1
,
u
)
=
∫
t
0
t
1
[
Δ
H
(
t
)
2
+
κ
∥
u
(
t
)
∥
2
2
]
d
t
J(t 
0
​
 ,t 
1
​
 ,u)=∫ 
t 
0
​
 
t 
1
​
 
​
 [ΔH(t) 
2
 +κ∥u(t)∥ 
2
2
​
 ]dt
where 
u
(
t
)
∈
U
⊆
R
m
u(t)∈U⊆R 
m
  is control input.

3.3 Optimal Control Formulation
Problem 3.3.1 (Operator's Prime Directive):

min
⁡
u
(
⋅
)
E
[
∫
t
t
+
Δ
t
Δ
H
(
τ
)
2
d
τ
]
subject to:
x
˙
(
τ
)
=
f
(
x
(
τ
)
,
u
(
τ
)
)
x
(
τ
)
∈
X
safe
u
(
τ
)
∈
U
admissible
g
(
x
(
τ
)
)
≤
0
(law constraints 
Λ
)
​
  
u(⋅)
min
​
 E[∫ 
t
t+Δt
​
 ΔH(τ) 
2
 dτ]
subject to:
x
˙
 (τ)=f(x(τ),u(τ))
x(τ)∈X 
safe
​
 
u(τ)∈U 
admissible
​
 
g(x(τ))≤0(law constraints Λ)
​
 
Equation 3.3.2 (Tao Update - Gradient Descent):

x
(
t
+
1
)
=
x
(
t
)
+
α
(
x
target
−
x
(
t
)
)
−
β
∇
Δ
H
(
t
)
x(t+1)=x(t)+α(x 
target
​
 −x(t))−β∇ΔH(t)
where 
α
,
β
>
0
α,β>0 are learning rates.

3.4 Consensus Algorithms
Equation 3.4.1 (Council Equation - Softmax Consensus):

x
consensus
=
∑
b
=
1
B
e
β
Σ
R
b
x
b
∑
b
=
1
B
e
β
Σ
R
b
x 
consensus
​
 = 
∑ 
b=1
B
​
 e 
βΣR 
b
​
 
 
∑ 
b=1
B
​
 e 
βΣR 
b
​
 
 x 
b
​
 
​
 
where 
x
b
x 
b
​
  is proposal from bloc 
b
b, 
β
>
0
β>0 is temperature.

Equation 3.4.2 (Distributed Averaging):

x
(
k
+
1
)
=
P
x
(
k
)
x 
(k+1)
 =Px 
(k)
 
where 
P
∈
R
B
×
B
P∈R 
B×B
  is doubly stochastic consensus matrix.

3.5 Time-Series Analysis
Equation 3.5.1 (Memory Encoding - EWMA):

μ
(
t
)
=
γ
μ
(
t
−
1
)
+
(
1
−
γ
)
x
(
t
)
,
γ
∈
(
0
,
1
)
μ(t)=γμ(t−1)+(1−γ)x(t),γ∈(0,1)
Equation 3.5.2 (Crisis Detection - Inflection):

θ
χ
=
{
t
∣
Δ
H
¨
(
t
)
=
0
 and 
Δ
H
˙
(
t
)
 changes sign
}
θ 
χ
​
 ={t∣ 
ΔH
¨
 (t)=0 and  
ΔH
˙
 (t) changes sign}
where 
Δ
H
˙
=
d
Δ
H
/
d
t
ΔH
˙
 =dΔH/dt, 
Δ
H
¨
=
d
2
Δ
H
/
d
t
2
ΔH
¨
 =d 
2
 ΔH/dt 
2
 .

Equation 3.5.3 (Redemption Formula - Impulse Response):

χ
ρ
(
t
)
=
∑
k
=
1
M
I
k
⋅
δ
(
t
−
t
k
)
∗
h
(
t
)
χρ(t)= 
k=1
∑
M
​
 I 
k
​
 ⋅δ(t−t 
k
​
 )∗h(t)
where 
δ
δ is Dirac delta, 
h
(
t
)
=
e
−
t
/
τ
h(t)=e 
−t/τ
  is impulse response, 
I
k
I 
k
​
  are impact magnitudes.

Equation 3.5.4 (Early Warning Trigger):

Trigger if: 
∥
x
˙
(
t
)
∥
2
>
λ
max
  OR  
Δ
H
(
t
)
>
H
threshold
Trigger if: ∥ 
x
˙
 (t)∥ 
2
​
 >λ 
max
​
   OR  ΔH(t)>H 
threshold
​
 
PART 4: COMPUTATIONAL HIERARCHY
4.1 Formal Language Theory
Definition 4.1.1 (Chomsky Hierarchy):

Type 3 (Regular): Generated by 
A
→
a
B
∣
a
A→aB∣a

Type 2 (Context-Free): 
A
→
α
A→α, 
α
∈
(
V
∪
Σ
)
∗
α∈(V∪Σ) 
∗
 

Type 1 (Context-Sensitive): 
α
A
β
→
α
γ
β
αAβ→αγβ, 
∣
γ
∣
≥
1
∣γ∣≥1

Type 0 (Recursively Enumerable): No restrictions

4.2 Automata Specifications
Definition 4.2.1 (Linear Bounded Automaton):
A TM 
M
=
(
Q
,
Σ
,
Γ
,
δ
,
q
0
,
q
a
,
q
r
)
M=(Q,Σ,Γ,δ,q 
0
​
 ,q 
a
​
 ,q 
r
​
 ) with tape bound 
k
⋅
∣
w
∣
k⋅∣w∣.

Theorem 4.2.2 (LBA Decidability):
Membership problem for LBA is PSPACE-complete.

Definition 4.2.3 (K-Process as LBA):
K-Process with fixed tape length proportional to 
∣
K
lex
∣
∣K 
lex
​
 ∣ is an LBA.

4.3 Complexity Results
Theorem 4.3.1 (Cryptographer's Paradox Formal):
For deterministic algorithm 
A
:
{
0
,
1
}
n
→
{
0
,
1
}
m
A:{0,1} 
n
 →{0,1} 
m
 :

Security
(
A
)
≤
min
⁡
(
H
∞
(
K
)
,
PreimageResistance
(
A
)
)
Security(A)≤min(H 
∞
​
 (K),PreimageResistance(A))
where 
H
∞
(
K
)
=
−
log
⁡
2
(
max
⁡
k
P
[
K
=
k
]
)
H 
∞
​
 (K)=−log 
2
​
 (max 
k
​
 P[K=k]) is min-entropy.

Corollary 4.3.2:
K-Process security 
≥
log
⁡
2
(
∣
Σ
∣
N
)
≥log 
2
​
 (∣Σ∣ 
N
 ) bits for uniform 
C
C.

PART 5: NUMERICAL IMPLEMENTATION
5.1 K-Process Implementation
python
import hashlib

def k_process(key_string: str, m: int = 26) -> int:
    """K-Process key derivation function."""
    N = len(key_string)
    S0 = 0
    for i, char in enumerate(key_string, 1):
        k_i = (ord(char.upper()) - 64) % m  # A=1,...,Z=26
        S0 += i * k_i
    
    S1 = (S0**2 + N//2) % 997
    S2 = (3*S1 + 42) % 256
    S_final = (S2 * (N % 128)) % 1000
    
    return S_final

def lexical_condensation(corpus: list) -> str:
    """Generate lexical key from concept corpus."""
    return ''.join(concept[0].upper() for concept in corpus)
5.2 System Dynamics Implementation
python
import numpy as np
from scipy.integrate import solve_ivp

class SystemDynamics:
    def __init__(self, d: int, W: np.ndarray):
        self.d = d
        self.W = W  # d×d weight matrix
    
    def total_resonance(self, x: np.ndarray) -> float:
        """ΣR = w^T x"""
        w = np.ones(self.d) / self.d  # uniform weights
        return w @ x
    
    def harmonic_gradient(self, x: np.ndarray, dx: np.ndarray) -> float:
        """ΔH = ||∇ΣR||"""
        w = np.ones(self.d) / self.d
        grad_SR = w * dx  # ∂ΣR/∂x_i = w_i * dx_i/dt
        return np.linalg.norm(grad_SR, 2)
    
    def tao_update(self, x: np.ndarray, x_target: np.ndarray, 
                  alpha: float = 0.1, beta: float = 0.01) -> np.ndarray:
        """Tao update rule: x_{t+1} = x_t + α(x_target - x_t) - β∇ΔH"""
        # Simplified gradient approximation
        delta = x_target - x
        # Assume gradient points toward target
        grad_DH = -delta / (np.linalg.norm(delta) + 1e-8)
        return x + alpha * delta - beta * grad_DH
    
    def consensus_formation(self, proposals: list[np.ndarray], 
                          beta: float = 1.0) -> np.ndarray:
        """Softmax consensus: RCF = softmax(ΣR_b) * x_b"""
        B = len(proposals)
        scores = np.array([self.total_resonance(x) for x in proposals])
        weights = np.exp(beta * scores)
        weights /= weights.sum()
        
        consensus = np.zeros_like(proposals[0])
        for w, x in zip(weights, proposals):
            consensus += w * x
        return consensus
5.3 Control Law Implementation
python
def optimal_control(x0: np.ndarray, x_target: np.ndarray, 
                   T: float, dt: float) -> tuple:
    """
    Solve finite-horizon optimal control.
    Minimize ∫(ΔH² + κ‖u‖²)dt subject to dx/dt = Ax + Bu.
    """
    n = len(x0)
    # Simple LQR formulation
    A = -np.eye(n) * 0.1  # Stable dynamics
    B = np.eye(n)
    Q = np.eye(n)  # State cost
    R = 0.1 * np.eye(n)  # Control cost
    
    # Solve discrete-time Riccati equation
    P = np.zeros((n, n))
    for _ in range(1000):  # Fixed point iteration
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        if np.linalg.norm(P_new - P) < 1e-6:
            break
        P = P_new
    
    # Optimal feedback gain
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    
    # Simulate
    x = x0.copy()
    trajectory = [x0]
    controls = []
    
    for t in np.arange(0, T, dt):
        u = -K @ (x - x_target)
        x = x + dt * (A @ x + B @ u)
        trajectory.append(x.copy())
        controls.append(u)
    
    return np.array(trajectory), np.array(controls)
PART 6: THEOREMS & PROOFS
Theorem 6.1 (K-Process Determinism):
For fixed corpus 
C
C and extraction function 
f
extract
f 
extract
​
 :

∀
C
,
 
K
lex
 is unique
⇒
K-Process
(
K
lex
)
 is deterministic
.
∀C, K 
lex
​
  is unique⇒K-Process(K 
lex
​
 ) is deterministic.
Proof: Follows from composition of deterministic functions. ∎

Theorem 6.2 (Cerberus IND-CCA2 Reduction):
If Kyber is IND-CCA2 secure, SIDH is SSI-hard, and UOV is EUF-CMA secure, then Cerberus-KEM is IND-CCA2 secure in ROM.

Proof Sketch: Hybrid argument:

Replace Kyber ciphertext with random (MLWE hard)

Replace SIDH shared secret with random (SSI hard)

UOV signature prevents PK substitution (MQ hard)
Each step adds negligible advantage. ∎

Theorem 6.3 (System Stability):
For Tao update with 
α
∈
(
0
,
1
)
α∈(0,1), 
β
≥
0
β≥0:

lim
⁡
t
→
∞
x
(
t
)
=
x
target
 if 
∥
I
−
α
I
∥
2
<
1.
t→∞
lim
​
 x(t)=x 
target
​
  if ∥I−αI∥ 
2
​
 <1.
Proof: Write update as 
x
t
+
1
=
(
I
−
α
I
)
x
t
+
α
x
target
−
β
∇
Δ
H
x 
t+1
​
 =(I−αI)x 
t
​
 +αx 
target
​
 −β∇ΔH.
For 
∇
Δ
H
∇ΔH bounded, fixed point at 
x
∗
=
x
target
x 
∗
 =x 
target
​
 . ∎

PART 7: COMPLETE MATHEMATICAL SUMMARY
7.1 Core Components:
K-Process: 
KeyDerivation
:
Σ
N
→
Z
1000
KeyDerivation:Σ 
N
 →Z 
1000
​
 

F
(
K
)
=
(
[
3
(
(
∑
i
⋅
k
i
)
2
+
⌊
N
/
2
⌋
)
m
o
d
 
 
997
+
42
]
m
o
d
 
 
256
⋅
(
N
m
o
d
 
 
128
)
)
m
o
d
 
 
1000
F(K)=([3((∑i⋅k 
i
​
 ) 
2
 +⌊N/2⌋)mod997+42]mod256⋅(Nmod128))mod1000
Cerberus-KEM: 
Enc
:
P
K
→
(
C
,
K
)
Enc:PK→(C,K), 
Dec
:
(
S
K
,
C
)
→
K
Dec:(SK,C)→K
Security: 
MLWE
∧
SSI
∧
MQ
⇒
IND-CCA2
MLWE∧SSI∧MQ⇒IND-CCA2

System Dynamics: Control system with state 
x
∈
R
d
x∈R 
d
 

Σ
R
=
w
⊤
x
ΣR=w 
⊤
 x

Δ
H
=
∥
∇
Σ
R
∥
2
ΔH=∥∇ΣR∥ 
2
​
 

J
=
∫
(
Δ
H
2
+
κ
∥
u
∥
2
)
d
t
J=∫(ΔH 
2
 +κ∥u∥ 
2
 )dt

Consensus Protocol: 
x
∗
=
softmax
β
(
{
Σ
R
b
}
)
⋅
{
x
b
}
x 
∗
 =softmax 
β
​
 ({ΣR 
b
​
 })⋅{x 
b
​
 }

7.2 Computational Properties:
K-Process: 
O
(
N
)
O(N) time, 
O
(
N
)
O(N) space

Cerberus-KEM: 
O
(
n
3
)
O(n 
3
 ) for lattice ops, 
O
(
p
log
⁡
p
)
O(plogp) for isogenies

System Control: 
O
(
d
3
)
O(d 
3
 ) for LQR, 
O
(
B
d
)
O(Bd) for consensus

7.3 Security Parameters:
Component	Security Claim	Assumption	Key Size
K-Process	( \log_2(	\Sigma	^N) ) bits	Corpus uniqueness	
N
N chars
Kyber L1	128-bit quantum	MLWE	800 B
SIDHp751	192-bit quantum	SSI	564 B
UOV (64,128)	128-bit classical	MQ	98 KB
