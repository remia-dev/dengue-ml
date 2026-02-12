package dengue.ml;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.optim.*;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;

import java.util.Arrays;

/**
 * Seasonal Autoregressive Integrated Moving Average (SARIMA) for predictive modeling.
 * <p>
 * Model: SARIMA(p,d,q)(P,D,Q)s
 * φ(B)Φ(B^s) ∇^d ∇_s^D y_t = θ(B)Θ(B^s) ε_t
 * <p>
 * - p,d,q: non-seasonal AR order, differencing, MA order
 * - P,D,Q: seasonal AR, seasonal differencing, seasonal MA
 * - s: season length (e.g. 12 for monthly data with yearly seasonality)
 */
public class Sarima {

    private final int p, d, q, P, D, Q, s;
    private final double[] ar;      // φ₁..φₚ
    private final double[] ma;     // θ₁..θq
    private final double[] seasonalAr; // Φ₁..Φₚ
    private final double[] seasonalMa; // Θ₁..ΘQ
    private final double[] differencedSeries;
    private final double intercept; // constant term (optional, often 0)
    private final int forecastStartIndex;

    public Sarima(int p, int d, int q, int P, int D, int Q, int s,
                  double[] data) {
        this.p = p;
        this.d = d;
        this.q = q;
        this.P = P;
        this.D = D;
        this.Q = Q;
        this.s = s;
        this.differencedSeries = difference(data);
        this.forecastStartIndex = differencedSeries.length;

        double[] params = estimateParameters(differencedSeries);
        int idx = 0;
        this.ar = new double[p];
        for (int i = 0; i < p; i++) ar[i] = params[idx++];
        this.ma = new double[q];
        for (int i = 0; i < q; i++) ma[i] = params[idx++];
        this.seasonalAr = new double[P];
        for (int i = 0; i < P; i++) seasonalAr[i] = params[idx++];
        this.seasonalMa = new double[Q];
        for (int i = 0; i < Q; i++) seasonalMa[i] = params[idx++];
        this.intercept = (params.length > idx) ? params[idx] : 0;
    }

    /** Apply non-seasonal differencing d times and seasonal differencing D times (lag s). */
    private double[] difference(double[] series) {
        double[] z = series.clone();
        for (int i = 0; i < d; i++) {
            z = diff(z, 1);
        }
        for (int i = 0; i < D; i++) {
            z = diff(z, s);
        }
        return z;
    }

    private static double[] diff(double[] x, int lag) {
        if (lag >= x.length) return new double[0];
        double[] out = new double[x.length - lag];
        for (int i = lag; i < x.length; i++) {
            out[i - lag] = x[i] - x[i - lag];
        }
        return out;
    }

    /** Estimate (φ, θ, Φ, Θ) using conditional sum of squares. */
    private double[] estimateParameters(double[] z) {
        int maxLag = p + s * P;
        int maxMaLag = q + s * Q;
        int start = Math.max(maxLag, maxMaLag);
        if (start >= z.length - 1) {
            return defaultParams(z);
        }

        int n = z.length - start;
        int nParams = p + q + P + Q;
        double[] best = defaultParams(z);
        double bestRss = Double.POSITIVE_INFINITY;

        try {
            MultivariateOptimizer opt = new BOBYQAOptimizer(Math.min(nParams + 2, nParams + 5));
            double[] lower = new double[nParams];
            double[] upper = new double[nParams];
            Arrays.fill(lower, -1.5);
            Arrays.fill(upper, 1.5);
            PointValuePair result = opt.optimize(
                new MaxEval(2000),
                new ObjectiveFunction(params -> conditionalSumOfSquares(z, params, start)),
                GoalType.MINIMIZE,
                new InitialGuess(best),
                new SimpleBounds(lower, upper)
            );
            bestRss = result.getValue();
            best = result.getPoint();
        } catch (Exception e) {
            // use defaults
        }
        return best;
    }

    private double conditionalSumOfSquares(double[] z, double[] params, int start) {
        int idx = 0;
        double[] arP = new double[p];
        for (int i = 0; i < p; i++) arP[i] = params[idx++];
        double[] maP = new double[q];
        for (int i = 0; i < q; i++) maP[i] = params[idx++];
        double[] sarP = new double[P];
        for (int i = 0; i < P; i++) sarP[i] = params[idx++];
        double[] smaP = new double[Q];
        for (int i = 0; i < Q; i++) smaP[i] = params[idx++];

        double[] innovations = new double[z.length];
        for (int t = start; t < z.length; t++) {
            double pred = 0;
            for (int i = 0; i < p && t - 1 - i >= 0; i++) pred += arP[i] * z[t - 1 - i];
            for (int i = 0; i < P && t - s * (i + 1) >= 0; i++) pred += sarP[i] * z[t - s * (i + 1)];
            for (int i = 0; i < q && t - 1 - i >= 0; i++) pred += maP[i] * innovations[t - 1 - i];
            for (int i = 0; i < Q && t - s * (i + 1) >= 0; i++) pred += smaP[i] * innovations[t - s * (i + 1)];
            innovations[t] = z[t] - pred;
        }
        double rss = 0;
        for (int t = start; t < z.length; t++) rss += innovations[t] * innovations[t];
        return rss;
    }

    private double[] defaultParams(double[] z) {
        double[] params = new double[p + q + P + Q];
        Arrays.fill(params, 0.1);
        if (p > 0 && z.length > p) {
            // Yule-Walker–style: regress z_t on z_{t-1}..z_{t-p}
            double[][] X = new double[z.length - p][p];
            double[] y = new double[z.length - p];
            for (int i = p; i < z.length; i++) {
                for (int j = 0; j < p; j++) X[i - p][j] = z[i - 1 - j];
                y[i - p] = z[i];
            }
            try {
                LinearRegression lr = new LinearRegression(X, y);
                for (int i = 0; i < p; i++) params[i] = lr.getCoefficient(i);
            } catch (Exception ignored) { }
        }
        return params;
    }

    /** Forecast the next `steps` values of the differenced series. */
    public double[] forecastDifferenced(int steps) {
        double[] z = Arrays.copyOf(differencedSeries, differencedSeries.length + steps);
        int T = differencedSeries.length;
        double[] innovations = new double[z.length];
        for (int t = T; t < T + steps; t++) {
            double pred = intercept;
            for (int i = 0; i < p && t - 1 - i >= 0; i++) pred += ar[i] * z[t - 1 - i];
            for (int i = 0; i < P && t - s * (i + 1) >= 0; i++) pred += seasonalAr[i] * z[t - s * (i + 1)];
            for (int i = 0; i < q && t - 1 - i >= 0; i++) pred += ma[i] * innovations[t - 1 - i];
            for (int i = 0; i < Q && t - s * (i + 1) >= 0; i++) pred += seasonalMa[i] * innovations[t - s * (i + 1)];
            z[t] = pred;
            innovations[t] = 0;
        }
        return Arrays.copyOfRange(z, T, T + steps);
    }

    /** Forecast in original scale by integrating differenced forecasts. Requires original series. */
    public double[] forecast(double[] originalSeries, int steps) {
        double[] diffForecast = forecastDifferenced(steps);
        if (d == 0 && D == 0) return diffForecast;
        double[] level = Arrays.copyOf(originalSeries, originalSeries.length + steps);
        int n = originalSeries.length;
        for (int i = 0; i < steps; i++) {
            double next = diffForecast[i];
            if (d > 0) next += level[n + i - 1];
            if (D > 0 && (n + i) >= s) {
                if (d > 0) next += level[n + i - s] - (n + i - s - 1 >= 0 ? level[n + i - s - 1] : 0);
                else next += level[n + i - s];
            }
            level[n + i] = next;
        }
        return Arrays.copyOfRange(level, n, n + steps);
    }

    /** When d=0 and D=0, forecast is in same scale as input. */
    public double[] forecast(int steps) {
        return forecastDifferenced(steps);
    }

    public int getSeasonLength() { return s; }
    public int getP() { return p; }
    public int getD() { return d; }
    public int getQ() { return q; }
    public int getSeasonalP() { return P; }
    public int getSeasonalD() { return D; }
    public int getSeasonalQ() { return Q; }
}
