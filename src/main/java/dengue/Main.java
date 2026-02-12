package dengue;

import dengue.ml.DenguePredictor;
import dengue.ml.LinearRegression;
import dengue.ml.Sarima;

import java.nio.file.Paths;

/**
 * Demo: linear regression (statistical modeling) and SARIMA (predictive modeling) for dengue.
 */
public class Main {

    public static void main(String[] args) {
        double[] cases = getSampleDengueCases();
        DenguePredictor predictor = new DenguePredictor(cases);

        // --- Linear regression: statistical relationship (e.g. cases vs time/season)
        predictor.fitLinearRegression();
        LinearRegression lr = predictor.getRegression();
        System.out.println("=== Linear regression (statistical modeling) ===");
        System.out.printf("Intercept β₀ = %.4f%n", lr.getIntercept());
        System.out.printf("Slope β₁ (time) = %.4f%n", lr.getCoefficient(0));
        System.out.printf("R² = %.4f, Adjusted R² = %.4f%n", lr.getRSquared(), lr.getAdjustedRSquared());
        double[] fitted = lr.predict(buildTimeCovariates(cases.length));
        System.out.println("Fitted (first 5): " + format(fitted, 5));
        System.out.println();

        // --- SARIMA: seasonal predictive model (e.g. monthly with s=12)
        int s = 12; // monthly data, yearly seasonality
        predictor.fitSarima(1, 0, 1, 1, 0, 1, s); // SARIMA(1,0,1)(1,0,1)12
        Sarima sarima = predictor.getSarima();
        System.out.println("=== SARIMA(1,0,1)(1,0,1)" + s + " (predictive modeling) ===");
        int steps = 6;
        double[] forecast = predictor.forecastSarima(steps);
        System.out.println("Forecast next " + steps + " periods: " + format(forecast, forecast.length));
        System.out.println();

        // Optional: run with CSV if path provided
        if (args.length > 0 && args[0] != null && !args[0].trim().isEmpty()) {
            try {
                DenguePredictor fromFile = DenguePredictor.fromCsv(Paths.get(args[0].trim()));
                fromFile.fitLinearRegression();
                fromFile.fitSarima(1, 0, 1, 1, 0, 1, 12);
                double[] fc = fromFile.forecastSarima(6);
                System.out.println("From CSV - Forecast: " + format(fc, fc.length));
            } catch (Exception e) {
                String msg = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
                System.err.println("CSV error: " + msg);
            }
        }
    }

    private static double[][] buildTimeCovariates(int n) {
        double[][] X = new double[n][1];
        for (int i = 0; i < n; i++) X[i][0] = i;
        return X;
    }

    private static String format(double[] a, int max) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < Math.min(a.length, max); i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.2f", a[i]));
        }
        if (a.length > max) sb.append("...");
        sb.append("]");
        return sb.toString();
    }

    /** Sample monthly dengue cases (synthetic) for demo. */
    private static double[] getSampleDengueCases() {
        return new double[] {
            45, 52, 61, 78, 88, 95, 102, 98, 85, 72, 58, 48,
            50, 55, 65, 82, 92, 100, 108, 104, 88, 75, 62, 51,
            48, 54, 68, 85, 94, 103, 112, 106, 90, 78, 64, 52,
            52, 58, 70, 86, 96, 105, 115, 108, 92, 80, 66, 55
        };
    }
}
