# The Procedure
When you trace out and calculate the derivatives one by one, the chain rule is bound to apply.

A good general rule, working backwards from the loss, is:
- Always include the derivative you just calculated (this is the chain rule)
- Think about the forward operations, then reverse it to get the order in which the operations should be inversed.

## Examples

```python
hpreact = bngain * bnraw + bnbias

# Non-linearity
h = torch.tanh(hpreact) # hidden layer
```

Note here that:
- `bngain` has a shape of (1, 64)
- `bnraw` has a shape of (32, 64)
- `bnbias` has a shape of (1, 64)
- `dh` has a shape of (32, 64)

### Ex. 1.

In the code snippet above, if we had to calculate the derivative for `bngain`, which would be `dbngain`, then:
1. Always include `dh`.
2. The order of operations in the forward pass:
    1. Broadcast `bngain`.
    2. Multiply by `bnraw`.
    3. Add `bnbias` (no effect here. Derivative would be 0)
    4. Assign to `hpreact`.

So then, we reverse the operations:
1. Assign to `hpreact` (equivalent to `dh`).
2. Add `bnbias` (again, that's 0).
3. Multiply by `bnraw`.
4. Broadcast `bngain` (this would be sum across the axis it was broadcasted on)

So our backward pass looks like:
```python
dbngain = (dh * bnraw).sum(0, keepdim=True)
```

### Ex. 2.
Using the same code above, we can try calculating the derivative of `bnraw`.

Again, the "forward" pass is the same, but we don't necessarily need everything:

1. Broadcast `bngain`. (no effect here, plus the shapes wouldn't match)
2. Multiply by `bngain`.
3. Add `bnbias` (no effect here. Derivative would be 0)
4. Assign to `hpreact`.

Then, reverse:

1. Assign to hpreact (`dh`).
2. Multiply by `bngain`.

So our backward pass looks like:
```python
dbnraw = dh * bngain
```

# Reverse Operations
- Broadcasting: Because broadcasting means to replicate a certain value over a dimension multiple times, and then using all of them, all those uses count as branches. 

- Sum over axis: 


# Finding the derivatives of a matrix multiplication

Given a simple matrix multiplication problem like described below, we can trace out the derivative of each of the components with respect to the output matrix (and its derivative).

$$\begin{align*}
\mathbf{D} &= \mathbf{A} \mathbf{B} + \mathbf{C} \\
\end{align*}
$$

In this case, $\mathbf{D}$ is the output matrix, $\mathbf{A}$ and $\mathbf{B}$ are to be matrix multiplied, and $\mathbf{C}$ is the bias matrix

Let's say they're all 2x2 matrices:

$$
\begin{bmatrix} d_{11} & d_{12} \\ d_{21} & d_{22} \end{bmatrix} = 
\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
\begin{bmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{bmatrix} +
\begin{bmatrix} c_{11} & c_{12} \\ c_{21} & c_{22} \end{bmatrix}
$$

Let's then type each element of d in terms of the elements of a, b, and c:
$$
\begin{align*}
d_{11} &= a_{11} b_{11} + a_{12} b_{21} + c_{11} \\
d_{12} &= a_{11} b_{12} + a_{12} b_{22} + c_{12} \\
d_{21} &= a_{21} b_{11} + a_{22} b_{21} + c_{21} \\
d_{22} &= a_{21} b_{12} + a_{22} b_{22} + c_{22} \\
\end{align*}
$$

## Derivation of Derivative With Respect to $\mathbf{A}$ (The `h` in our case)

Now, in our case, we want to find the derivative of the loss with respect to each of the elements in $\mathbf{A}$. The derivative with respect to each element in $\mathbf{A}$, in our case, is:

$$
\begin{align*}
\frac{\partial L}{\partial a_{xy}} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial a_{xy}}
\end{align*}
$$

Here, we're not quite being precise with our definition, but we can trace out the derivative with respect to the first element of $\mathbf{A}$, $a_{11}$, as an example.

We basically look at where any $a_{11}$ appears in the equations, which are $d_{11}$ and $d_{12}$:

$$
\begin{align*}
d_{11} &= a_{11} b_{11} + a_{12} b_{21} + c_{11} \\
d_{12} &= a_{11} b_{12} + a_{12} b_{22} + c_{12} \\
\end{align*}
$$

The derivative of each, then, would be:

$$
\begin{align*}
\frac{\partial d_{11}}{\partial a_{11}} &= b_{11} \\
\frac{\partial d_{12}}{\partial a_{11}} &= b_{12}
\end{align*}
$$

Since there are multiple paths from $a_{11}$ to $d$, we have to sum over all of them:
$$
\begin{align*}
\frac{\partial d}{\partial a_{11}} &= \frac{\partial d_{11}}{\partial a_{11}} + \frac{\partial d_{12}}{\partial a_{11}} \\
&= b_{11} + b_{12}
\end{align*}
$$

Then, we can apply the chain rule to find the derivative of the loss with respect to $a_{11}$:

$$
\begin{align*}
\frac{\partial L}{\partial a_{11}} &= \frac{\partial L}{\partial d_{11}} \cdot \frac{\partial d_{11}}{\partial a_{11}} + \frac{\partial L}{\partial d_{12}} \cdot \frac{\partial d_{12}}{\partial a_{11}} \\
&= \frac{\partial L}{\partial d_{11}} \cdot b_{11} + \frac{\partial L}{\partial d_{12}} \cdot b_{12}
\end{align*}
$$

Thus, for each of the elements of $a$, we get these:

$$
\begin{align*}
\frac{\partial L}{\partial a_{11}} &= \frac{\partial L}{\partial d_{11}} \cdot b_{11} + \frac{\partial L}{\partial d_{12}} \cdot b_{12} \\
\frac{\partial L}{\partial a_{12}} &= \frac{\partial L}{\partial d_{11}} \cdot b_{21} + \frac{\partial L}{\partial d_{12}} \cdot b_{22} \\
\frac{\partial L}{\partial a_{21}} &= \frac{\partial L}{\partial d_{21}} \cdot b_{11} + \frac{\partial L}{\partial d_{22}} \cdot b_{12} \\
\frac{\partial L}{\partial a_{22}} &= \frac{\partial L}{\partial d_{21}} \cdot b_{21} + \frac{\partial L}{\partial d_{22}} \cdot b_{22} \\
\end{align*}
$$

Now, our $d$ is `logits`, so its derivative is `dlogits`; $b$ is `weights`, and $a$ is `h`, whose derivative is `dh`. To prevent us from being too verbose, we'll rewrite:
- `dlogits` as `l`
- `weights` as `w`

 so we can rewrite the above as:
$$
\begin{align*}
dh_{11} &= l_{11} \cdot w_{11} + l_{12} \cdot w_{12} \\
dh_{12} &= l_{11} \cdot w_{21} + l_{12} \cdot w_{22} \\
dh_{21} &= l_{21} \cdot w_{11} + l_{22} \cdot w_{12} \\
dh_{22} &= l_{21} \cdot w_{21} + l_{22} \cdot w_{22} \\
\end{align*}
$$

This looks like a matrix multiplication! Let's see what each $dh$ would be:

$$
\begin{align*}
dh_{11} &= \begin{bmatrix} l_{11} & l_{12} \end{bmatrix} \begin{bmatrix} w_{11} \\ w_{12} \end{bmatrix} \\
dh_{12} &= \begin{bmatrix} l_{11} & l_{12} \end{bmatrix} \begin{bmatrix} w_{21} \\ w_{22} \end{bmatrix} \\
dh_{21} &= \begin{bmatrix} l_{21} & l_{22} \end{bmatrix} \begin{bmatrix} w_{11} \\ w_{12} \end{bmatrix} \\
dh_{22} &= \begin{bmatrix} l_{21} & l_{22} \end{bmatrix} \begin{bmatrix} w_{21} \\ w_{22} \end{bmatrix} \\
\end{align*}
$$

Conveniently, we can write this in matrix form as:
$$
\begin{bmatrix} dh_{11} & dh_{12} \\ dh_{21} & dh_{22} \end{bmatrix} =
\begin{bmatrix} l_{11} & l_{12} \\ l_{21} & l_{22} \end{bmatrix}
\begin{bmatrix} w_{11} & w_{21} \\ w_{12} & w_{22} \end{bmatrix}
$$

Notice how $\mathbf{w}$ is transposed! This means the above can be rewritten as:
`dh = dlogits @ weights.T`.

This makes total sense as:
- `dh` has a shape of (32, 64)
- `dlogits` has a shape of (32, 27)
- `weights` has a shape of (64, 27)
- `weights.T` has a shape of (27, 64)
- `dlogits @ weights.T` is multiplying the shapes (32, 27) and (27, 64) together, which turns out to be (32, 64), and is the shape of `dh`.

## Derivation of Derivative With Respect to $\mathbf{B}$ (The `weights` in our case)

### Guesswork
Following the last lines of reasoning from the section above, can we hypothesize the formula for finding the derivative of the weights?

- `dw` has/needs a shape of (64, 27) (same as `weights`)
- `dlogits` has a shape of (32, 27)
- `h` has a shape of (32, 64)
- Could it be that we just do `h.T @ dlogits`? Or conversely, `(dlogits.T @ h).T`?

Turns out `h.T @ dlogits` is the answer! For some reason, `(dlogits.T @ h).T` is just a tiny little bit imprecise. Let's see how we can derive it.


### Derivation
Let's remind ourselves of the equations for $d$:
$$
\begin{align*}
d_{11} &= a_{11} b_{11} + a_{12} b_{21} + c_{11} \\
d_{12} &= a_{11} b_{12} + a_{12} b_{22} + c_{12} \\
d_{21} &= a_{21} b_{11} + a_{22} b_{21} + c_{21} \\
d_{22} &= a_{21} b_{12} + a_{22} b_{22} + c_{22} \\
\end{align*}
$$

Similar to how we derived the derivative with respect to $\mathbf{A}$, we can derive the derivative with respect to $\mathbf{B}$ by looking at where each element of $\mathbf{B}$ appears in the equations for $d$. I won't bore you with the steps to derive them. Let's just write them out:

$$
\begin{align*}
dh_{11} &= l_{11} \cdot a_{11} + l_{21} \cdot a_{21} \\
dh_{12} &= l_{12} \cdot a_{11} + l_{22} \cdot a_{21} \\
dh_{21} &= l_{11} \cdot a_{12} + l_{21} \cdot a_{22} \\
dh_{22} &= l_{12} \cdot a_{12} + l_{22} \cdot a_{22} \\
\end{align*}
$$

We can then write this in the vector form (replacing $a$ with $h$):
$$
\begin{align*}
dh_{11} &= \begin{bmatrix} l_{11} & l_{21} \end{bmatrix} \begin{bmatrix} h_{11} \\ h_{21} \end{bmatrix} \\
dh_{12} &= \begin{bmatrix} l_{12} & l_{22} \end{bmatrix} \begin{bmatrix} h_{11} \\ h_{21} \end{bmatrix} \\
dh_{21} &= \begin{bmatrix} l_{11} & l_{21} \end{bmatrix} \begin{bmatrix} h_{12} \\ h_{22} \end{bmatrix} \\
dh_{22} &= \begin{bmatrix} l_{12} & l_{22} \end{bmatrix} \begin{bmatrix} h_{12} \\ h_{22} \end{bmatrix} \\
\end{align*}
$$

That means we can write the above in matrix form:
$$
\begin{bmatrix} dh_{11} & dh_{12} \\ dh_{21} & dh_{22} \end{bmatrix} =
\begin{bmatrix} l_{11} & l_{21} \\ l_{12} & l_{22} \end{bmatrix}
\begin{bmatrix} h_{11} & h_{12} \\ h_{21} & h_{22} \end{bmatrix}
$$

which is `dh = dlogits.T @ h`. But again, because this example is just a simple 2x2, we aren't accounting for other possible sizes of matrices. As mentioned in the Guesswork section above, rearranging our equation will give us a better result: `dh = h.T @ dlogits`.

## Derivation of Derivative With Respect to $\mathbf{C}$ (The `bias` in our case)

Our bias matrix was broadcasted from a shape of (27) into (32, 27) in order to be added to the output of the matrix multiplication.

This means our $\mathbf{C}$ looks like this:
$$\begin{bmatrix} c_{1} & c_{2} & ... & c_{27} \\ c_{1} & c_{2} & ... & c_{27} \\ c_{1} & c_{2} & ... & c_{27} \\ \vdots & \vdots & \ddots & \vdots \\ c_{1} & c_{2} & ... & c_{27} \end{bmatrix}$$

The derivative of $\mathbf{C}$ with respect to the loss is:
$$
\begin{align*}
\frac{\partial{L}}{\partial{c_{1}}} &= \frac{\partial{d_{11}}}{\partial{c_{1}}} \cdot \frac{\partial{L}}{\partial{d_{11}}} + \frac{\partial{d_{21}}}{\partial{c_{1}}} \cdot \frac{\partial{L}}{\partial{d_{21}}} \\
\frac{\partial{L}}{\partial{c_{2}}} &= \frac{\partial{d_{12}}}{\partial{c_{2}}} \cdot \frac{\partial{L}}{\partial{d_{12}}} + \frac{\partial{d_{22}}}{\partial{c_{2}}} \cdot \frac{\partial{L}}{\partial{d_{22}}} \\
\end{align*}
$$

And since the partial derivatives of $d$ are:
$$
\begin{align*}
\frac{\partial{d_{11}}}{\partial{c_{1}}} &= 1 \\
\frac{\partial{d_{12}}}{\partial{c_{2}}} &= 1 \\
\frac{\partial{d_{21}}}{\partial{c_{1}}} &= 1 \\
\frac{\partial{d_{22}}}{\partial{c_{2}}} &= 1 \\
\end{align*}
$$

Since $c_x$ is just added.

That makes our derivative of $\mathbf{C}$ with respect to the loss:

$$
\begin{align*}
\frac{\partial{L}}{\partial{c_{1}}} &= 1 \cdot \frac{\partial{L}}{\partial{d_{11}}} + 1 \cdot \frac{\partial{L}}{\partial{d_{21}}} \\
\frac{\partial{L}}{\partial{c_{2}}} &= 1 \cdot \frac{\partial{L}}{\partial{d_{12}}} + 1 \cdot \frac{\partial{L}}{\partial{d_{22}}} \\
\end{align*}
$$

Which, in our terms, to rewrite $c$ as $b$ (for bias), and again $d$ as $l$ (the derivative of logits), is simply:
$$
\begin{align*}
db_{1} &= l_{11} + l_{21} \\
db_{2} &= l_{12} + l_{22} \\
\end{align*}
$$

This means we simply sum up each column of `dlogits`!
```python
db = dlogits.sum(dim=0)
```

## The cheat code
The derivations are fun, but also long and tedious. Once you arrive at the conclusions above, you'll realize that the shapes just have to work out! 

> 💡 If you know the shapes of the matrices, you can just guess the formula for the derivatives by ensuring that the shapes work out.

This is a much faster way to arrive at the same conclusions as above.

Since chain rule is involved, you do have to include something like `dlogits`. When you know that that's involved, you can guess which other matrix or vector you need in order to make the shapes work out!...