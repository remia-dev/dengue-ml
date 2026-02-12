package dengue.ml;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Combines linear regression (statistical modeling) and SARIMA (predictive modeling)
 * for dengue case analysis and forecasting.
 */
public class DenguePredictor {

    /** Historical dengue cases (e.g. monthly counts). */
    private final double[] cases;
    /** Optional covariates for linear regression: e.g. month, rainfall, temperature (row per period). */
    private final double[][] covariates;
    private LinearRegression regression;
    private Sarima sarima;

    public DenguePredictor(double[] cases, double[][] covariates) {
        if (cases == null || cases.length == 0) throw new IllegalArgumentException("cases required");
        this.cases = cases;
        this.covariates = (covariates != null && covariates.length == cases.length) ? covariates : null;
    }

    public DenguePredictor(double[] cases) {
        this(cases, null);
    }

    /** Fit linear regression of cases on covariates (e.g. time, season). */
    public void fitLinearRegression() {
        if (covariates == null) {
            double[][] timeOnly = new double[cases.length][1];
            for (int i = 0; i < cases.length; i++) timeOnly[i][0] = i;
            regression = new LinearRegression(timeOnly, cases);
        } else {
            regression = new LinearRegression(covariates, cases);
        }
    }

    /**
     * Fit SARIMA(p,d,q)(P,D,Q)s for seasonal dengue series.
     * Typical for monthly data with yearly seasonality: s=12, e.g. (1,0,1)(1,0,1)12.
     */
    public void fitSarima(int p, int d, int q, int P, int D, int Q, int s) {
        sarima = new Sarima(p, d, q, P, D, Q, s, cases);
    }

    public LinearRegression getRegression() { return regression; }
    public Sarima getSarima() { return sarima; }
    public double[] getCases() { return cases; }

    /** Predict next steps using SARIMA (in original case-count scale). */
    public double[] forecastSarima(int steps) {
        if (sarima == null) throw new IllegalStateException("Fit SARIMA first");
        if (sarima.getD() > 0 || sarima.getSeasonalD() > 0) {
            return sarima.forecast(cases, steps);
        }
        return sarima.forecast(steps);
    }

    /** Load dengue cases (and optional covariates) from a CSV. Expected: one row per period, first column = cases. */
    public static DenguePredictor fromCsv(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path);
        if (lines.isEmpty()) throw new IllegalArgumentException("Empty file");
        List<Double> caseList = new ArrayList<>();
        List<double[]> covList = new ArrayList<>();
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] parts = line.split("[,;\t]+");
            if (parts.length < 1) continue;
            try {
                double c = Double.parseDouble(parts[0].trim());
                caseList.add(c);
                if (parts.length > 1) {
                    double[] row = new double[parts.length - 1];
                    for (int j = 1; j < parts.length; j++) row[j - 1] = Double.parseDouble(parts[j].trim());
                    covList.add(row);
                }
            } catch (NumberFormatException e) {
                continue; // skip header or invalid lines
            }
        }
        double[] casesArr = caseList.stream().mapToDouble(Double::doubleValue).toArray();
        double[][] cov = (covList.size() == casesArr.length) ? covList.toArray(new double[0][]) : null;
        return new DenguePredictor(casesArr, cov);
    }
}
