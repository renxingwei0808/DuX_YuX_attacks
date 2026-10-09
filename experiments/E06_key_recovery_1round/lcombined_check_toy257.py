# Stand-alone check script: toy DuX(257), 7 rounds, real round order. Test A = L-combined equations [L0*sum SL(P+K)]_{4b+3}=0; Test B = r_KR=1 key recovery via y3-type equations (W position 3), plus a negative control.
# End-to-end check on toy DuX(257), 7 rounds, CC model, s=2 structures (active words 2,6):
#  A) L-combined equations  [L0 * sum_C SL(P_C+K)]_{4b+3} = 0 (uses only position-3 balance)
#  B) r_KR=1 master-key recovery from y0-type equations (19 unknowns/block), no key knowledge used.
import numpy as np, itertools, sympy as sp, random, time
p=257; A=179%p
ROT0=[0,1,3,4,7,8,9,10,12,14,15]
def S_fwd(x):
    x0,x1,x2,x3=x
    y2=(x2-x0*x3-A)%p; y1=(x1-x3*y2-A)%p; y0=(x0-y1*y2-A)%p; y3=(x3-y0*y1-A)%p
    return [y0,y1,y2,y3]
def SL(X): 
    Y=[]
    for b in range(4): Y+=S_fwd(X[4*b:4*b+4])
    return Y
import sympy as _sp
_Minv=_sp.Matrix(16,16,lambda i,j: 1 if ((j-i)%16) in (1,4,8,9,13) else 0)   # rows: Y_i = sum_j M[i][j] X_j
_M0=_Minv.inv_mod(p)
M0=[[int(_M0[i,j])%p for j in range(16)] for i in range(16)]
def L0(X): return [sum(M0[i][j]*X[j] for j in range(16))%p for i in range(16)]
def Lt(X,t):  # L_t = Rot_{-4t} o L0
    Y=L0(X); return [Y[(i-4*t)%16] for i in range(16)]
def txor(rk): return (rk[15]&1)^((rk[15]>>1)&1)
def enc(P,rks):   # rks[0..r]
    r=len(rks)-1
    X=[(P[i]+rks[0][i])%p for i in range(16)]
    for i in range(1,r+1):
        X=SL(X)
        if i<r: X=Lt(X,txor(rks[i])); X=[(X[j]+rks[i][j])%p for j in range(16)]
        else:   X=[(X[j]+rks[i][j])%p for j in range(16)]
    return X
# decryption (vectorised)
def R(x):
    x0,x1,x2,x3=x; return [x1,x2,(x0*x1+x3+A)%p,x0]
def SLinv(X):
    Y=[]
    for b in range(4):
        blk=X[4*b:4*b+4]
        for _ in range(4): blk=R(blk)
        Y+=blk
    return Y
def Linv(X,t):
    Y=[(X[(i+1)%16]+X[(i+4)%16]+X[(i+8)%16]+X[(i+9)%16]+X[(i+13)%16])%p for i in range(16)]
    return [Y[(i+4*t)%16] for i in range(16)]
def dec(C,rks):
    r=len(rks)-1; X=C
    for i in range(r,0,-1):
        X=[(X[j]-rks[i][j])%p for j in range(16)]
        if i<r: X=Linv(X,txor(rks[i]))
        X=SLinv(X)
    return [(X[j]-rks[0][j])%p for j in range(16)]
random.seed(7)
rks=[[random.randrange(p) for _ in range(16)] for _ in range(8)]  # 7 rounds -> rk^0..rk^7
# sanity: dec(enc)=id
P=[random.randrange(p) for _ in range(16)]
Pn=[np.array([v]) for v in P]
assert [int(v[0]) for v in dec([np.array([c]) for c in enc(P,rks)],rks)]==P
print("enc/dec consistent")
# symbolic y0,y3 expansion -> unknown monomials
ks=sp.symbols('k0:4'); ps=sp.symbols('p0:4'); a=sp.Symbol('alpha')
def S_sym(x):
    x0,x1,x2,x3=x
    y2=sp.expand(x2-x0*x3-a); y1=sp.expand(x1-x3*y2-a); y0=sp.expand(x0-y1*y2-a); y3=sp.expand(x3-y0*y1-a)
    return [y0,y1,y2,y3]
Ssym=S_sym([ps[i]+ks[i] for i in range(4)])
def eq_template(c):
    poly=sp.Poly(Ssym[c].subs(a,A),*ks)   # coefficients are polys in p0..p3
    terms=[]  # (kexp, coeff poly in p)
    for kexp,coef in poly.terms():
        terms.append((kexp,sp.Poly(coef,*ps)))
    return terms
T0=eq_template(0); unk=[e for e,_ in T0 if any(e)]
print("y0 template: k-monomials =",len(unk))
def structure(active):
    N=p*p
    g0,g1=np.meshgrid(np.arange(p),np.arange(p),indexing='ij')
    consts=[random.randrange(p) for _ in range(16)]
    C=[np.full(N,c,dtype=np.int64) for c in consts]
    C[active[0]]=g0.reshape(-1); C[active[1]]=g1.reshape(-1)
    return dec(C,rks)

def pow_arr(arr,e):
    out=np.ones_like(arr)
    for _ in range(e): out=(out*arr)%p
    return out
def build_rows(Pc,ctype):
    T=eq_template(ctype); unk=[e for e,_ in T if any(e)]
    out=[]
    for b in range(4):
        pw=Pc[4*b:4*b+4]; cache={}
        def psum(e):
            if e not in cache:
                m=np.ones(len(pw[0]),dtype=np.int64)
                for i in range(4): m=(m*pow_arr(pw[i],e[i]))%p
                cache[e]=int(m.sum()%p)
            return cache[e]
        row={}; const=0
        for kexp,cp in T:
            val=sum(int(c)*psum(pexp) for pexp,c in cp.terms())%p
            if any(kexp): row[kexp]=val
            else: const=val
        out.append((row,(-const)%p))
    return out
def solve_mod(Arows,brhs):
    M=[r[:]+[v] for r,v in zip(Arows,brhs)]; n=len(Arows[0]); m=len(M); piv=[]; r=0
    for c in range(n):
        pr=next((i for i in range(r,m) if M[i][c]%p),None)
        if pr is None: continue
        M[r],M[pr]=M[pr],M[r]; inv=pow(M[r][c],p-2,p); M[r]=[(x*inv)%p for x in M[r]]
        for i in range(m):
            if i!=r and M[i][c]%p:
                f=M[i][c]; M[i]=[(x-f*y)%p for x,y in zip(M[i],M[r])]
        piv.append(c); r+=1
    return M,piv,r
def structure(active,rks):
    s=len(active); N=p**s
    grids=np.meshgrid(*[np.arange(p)]*s,indexing='ij')
    consts=[random.randrange(p) for _ in range(16)]
    C=[np.full(N,c,dtype=np.int64) for c in consts]
    for g,a in zip(grids,active): C[a]=g.reshape(-1)
    return dec(C,rks)
def attack(rounds,active,ctypes,NS,seed):
    random.seed(seed)
    rks=[[random.randrange(p) for _ in range(16)] for _ in range(rounds+1)]
    t0=time.time(); eqs={b:[] for b in range(4)}
    for s in range(NS):
        Pc=structure(active,rks)
        for ct in ctypes:
            for b,(row,rhs) in enumerate(build_rows(Pc,ct)): eqs[b].append((row,rhs))
    ok=True
    for b in range(4):
        unk=sorted({k for row,_ in eqs[b] for k in row},key=lambda t:(sum(t),t))
        A=[[row.get(u,0) for u in unk] for row,_ in eqs[b]]; rhs=[r for _,r in eqs[b]]
        M,piv,rank=solve_mod(A,rhs)
        # consistency: rows beyond rank must have rhs 0
        consistent=all(M[i][-1]%p==0 for i in range(rank,len(M)))
        sol={unk[c]:M[i][-1] for i,c in enumerate(piv)}
        lin=[sol.get(tuple(1 if j==i else 0 for j in range(4))) for i in range(4)]
        good=(lin==rks[0][4*b:4*b+4])
        ok&=good
        print(f"  block {b}: unknowns {len(unk)} rank {rank} consistent={consistent} k={lin} true={rks[0][4*b:4*b+4]} {'OK' if good else 'MISMATCH'}")
    print(f"  -> {rounds} rounds, active {active}, eq types {ctypes}, {NS} structures: {'SUCCESS' if ok else 'FAIL'} ({time.time()-t0:.1f}s)")
print("B1) 6-round toy-257, s=1 (Z_2 all positions balanced) -> y0-type equations")
attack(6,(2,),[0],22,11)
print("B1') 6-round, y2-type equations (2 unknowns)")
attack(6,(2,),[2],4,12)
print("B2) 7-round toy-257, s=2 (Z_2 balanced at {0,3}) -> y3-type equations (W position 3 only)")
attack(7,(2,6),[3],72,13)
print("B2-neg) 7-round, s=2, y0-type equations (W position 0 NOT available) -> expected to fail")
attack(7,(2,6),[0],24,14)
