package dengue.ml;

import org.apache.commons.math3.linear.*;

/**
 * Ordinary Least Squares (OLS) linear regression for statistical modeling.
 * <p>
 * Model: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
 * <p>
 * Closed-form solution (normal equation): β = (X'X)⁻¹X'y
 * where X is the design matrix (with column of 1s for intercept) and y is the response vector.
 */
public class LinearRegression {

    private final double[] coefficients;  // β₀, β₁, ..., βₙ
    private double rSquared;
    private double adjustedRSquared;
    private int n;
    private int p;

    /**
     * Fit the model using the normal equation: β = (X'X)⁻¹X'y
     *
     * @param X design matrix (rows = observations, columns = features; no intercept column)
     * @param y response vector (length = number of observations)
     */
    public LinearRegression(double[][] X, double[] y) {
        if (X == null || y == null || X.length != y.length || X.length == 0) {
            throw new IllegalArgumentException("X and y must be non-null, same length, and non-empty");
        }
        n = X.length;
        int features = X[0].length;
        p = features + 1; // +1 for intercept

        // Build design matrix with intercept column (1s)
        double[][] design = new double[n][p];
        for (int i = 0; i < n; i++) {
            design[i][0] = 1.0;
            for (int j = 0; j < features; j++) {
                design[i][j + 1] = X[i][j];
            }
        }

        RealMatrix Xm = MatrixUtils.createRealMatrix(design);
        RealVector yv = MatrixUtils.createRealVector(y);

        // β = (X'X)⁻¹ X' y
        RealMatrix Xt = Xm.transpose();
        RealMatrix XtX = Xt.multiply(Xm);
        DecompositionSolver solver = new LUDecomposition(XtX).getSolver();
        if (!solver.isNonSingular()) {
            throw new IllegalArgumentException("Design matrix X'X is singular; cannot compute (X'X)⁻¹");
        }
        RealVector beta = solver.solve(Xt.operate(yv));
        coefficients = beta.toArray();

        // R² = 1 - SS_res / SS_tot
        double[] fitted = predict(X);
        double meanY = 0;
        for (double v : y) meanY += v;
        meanY /= n;
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++) {
            ssTot += (y[i] - meanY) * (y[i] - meanY);
            ssRes += (y[i] - fitted[i]) * (y[i] - fitted[i]);
        }
        rSquared = (ssTot > 0) ? 1.0 - (ssRes / ssTot) : 0;
        adjustedRSquared = (n > p) ? 1.0 - (1.0 - rSquared) * (n - 1) / (n - p) : rSquared;
    }

    /** Intercept β₀ */
    public double getIntercept() {
        return coefficients[0];
    }

    /** Coefficient βᵢ for feature i (0-based). β₁ is first feature. */
    public double getCoefficient(int i) {
        return coefficients[i + 1];
    }

    /** All coefficients [β₀, β₁, ..., βₙ] */
    public double[] getCoefficients() {
        return coefficients.clone();
    }

    public double getRSquared() { return rSquared; }
    public double getAdjustedRSquared() { return adjustedRSquared; }

    /** Predict y for one observation (no intercept in x). */
    public double predict(double[] x) {
        double y = coefficients[0];
        for (int i = 0; i < x.length; i++) {
            y += coefficients[i + 1] * x[i];
        }
        return y;
    }

    /** Predict y for each row of X. */
    public double[] predict(double[][] X) {
        double[] out = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            out[i] = predict(X[i]);
        }
        return out;
    }
}
