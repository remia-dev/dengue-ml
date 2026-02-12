package dengue;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import dengue.ml.DenguePredictor;
import dengue.ml.LinearRegression;
import io.javalin.Javalin;
import io.javalin.http.staticfiles.Location;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Web app for dengue ML: linear regression (statistical) + SARIMA (predictive).
 * Run with: mvn exec:java -Dexec.mainClass="dengue.WebApp"
 * Open http://localhost:7000 (or http://127.0.0.1:7000)
 */
public class WebApp {

    private static final Gson GSON = new Gson();

    private static int getPort() {
        String env = System.getenv("PORT");
        if (env != null && !env.isBlank()) {
            try {
                return Integer.parseInt(env.trim());
            } catch (NumberFormatException ignored) { }
        }
        return 7000;
    }

    public static void main(String[] args) {
        int port = getPort();
        Javalin app = Javalin.create(cfg -> {
            cfg.staticFiles.add("/public", Location.CLASSPATH);
        }).start("0.0.0.0", port);

        // Serve index from classpath so it always works (avoids static path issues)
        app.get("/", ctx -> {
            String html = loadIndexHtml();
            ctx.contentType("text/html").result(html);
        });
        app.get("/index.html", ctx -> {
            String html = loadIndexHtml();
            ctx.contentType("text/html").result(html);
        });

        app.post("/api/analyze", ctx -> {
            Map<String, Object> out = new HashMap<>();
            try {
                String body = ctx.body();
                if (body == null || body.isBlank()) {
                    out.put("error", "Missing request body");
                    sendJson(ctx, 200, out);
                    return;
                }
                Type type = new TypeToken<Map<String, Object>>() {}.getType();
                Map<String, Object> req = GSON.fromJson(body, type);
                if (req == null) {
                    out.put("error", "Invalid JSON");
                    sendJson(ctx, 200, out);
                    return;
                }
                Object casesObj = req.get("cases");
                if (!(casesObj instanceof List<?>)) {
                    out.put("error", "Missing or invalid 'cases' array");
                    sendJson(ctx, 200, out);
                    return;
                }
                List<?> casesList = (List<?>) casesObj;
                if (casesList.isEmpty()) {
                    out.put("error", "Empty 'cases' array");
                    sendJson(ctx, 200, out);
                    return;
                }
                double[] cases = casesList.stream()
                    .mapToDouble(o -> o instanceof Number ? ((Number) o).doubleValue() : Double.NaN)
                    .filter(d -> !Double.isNaN(d))
                    .toArray();
                if (cases.length != casesList.size()) {
                    out.put("error", "All case values must be numbers");
                    sendJson(ctx, 200, out);
                    return;
                }

                int p = 1, d = 0, q = 1, P = 1, D = 0, Q = 1, s = 12;
                int forecastSteps = 6;
                if (req.containsKey("sarima")) {
                    @SuppressWarnings("unchecked")
                    Map<String, Number> sarima = (Map<String, Number>) req.get("sarima");
                    if (sarima != null) {
                        p = getInt(sarima, "p", p);
                        d = getInt(sarima, "d", d);
                        q = getInt(sarima, "q", q);
                        P = getInt(sarima, "P", P);
                        D = getInt(sarima, "D", D);
                        Q = getInt(sarima, "Q", Q);
                        s = getInt(sarima, "s", s);
                    }
                }
                if (req.containsKey("forecastSteps")) {
                    Object fs = req.get("forecastSteps");
                    if (fs instanceof Number) {
                        forecastSteps = ((Number) fs).intValue();
                        forecastSteps = Math.max(1, Math.min(forecastSteps, 60));
                    }
                }

                DenguePredictor predictor = new DenguePredictor(cases);
                predictor.fitLinearRegression();
                predictor.fitSarima(p, d, q, P, D, Q, s);

                LinearRegression lr = predictor.getRegression();
                Map<String, Object> regression = new HashMap<>();
                regression.put("intercept", lr.getIntercept());
                regression.put("rSquared", lr.getRSquared());
                regression.put("adjustedRSquared", lr.getAdjustedRSquared());
                double[] coef = lr.getCoefficients();
                List<Double> coefList = new ArrayList<>(coef.length);
                for (double c : coef) coefList.add(c);
                regression.put("coefficients", coefList);
                // Fitted values for plotting: y_hat = X * beta (time-only design matrix)
                double[][] timeOnly = new double[cases.length][1];
                for (int i = 0; i < cases.length; i++) timeOnly[i][0] = i;
                double[] fitted = lr.predict(timeOnly);
                List<Double> fittedList = new ArrayList<>(fitted.length);
                for (double v : fitted) fittedList.add(v);
                regression.put("fitted", fittedList);
                out.put("regression", regression);

                double[] forecast = predictor.forecastSarima(forecastSteps);
                List<Double> forecastList = new ArrayList<>(forecast.length);
                for (double v : forecast) forecastList.add(v);
                out.put("forecast", forecastList);
            } catch (Throwable e) {
                String msg = e.getMessage();
                out.put("error", msg != null && !msg.isEmpty() ? msg : e.getClass().getSimpleName());
            }
            sendJson(ctx, 200, out);
        });

        app.get("/api/sample", ctx -> {
            try {
                double[] raw = sampleDengueCases();
                List<Double> list = new ArrayList<>(raw.length);
                for (double v : raw) list.add(v);
                sendJson(ctx, 200, list);
            } catch (Throwable e) {
                Map<String, Object> err = new HashMap<>();
                err.put("error", e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName());
                sendJson(ctx, 200, err);
            }
        });

        app.get("/api/health", ctx -> {
            Map<String, Object> h = new HashMap<>();
            h.put("status", "ok");
            h.put("port", port);
            sendJson(ctx, 200, h);
        });

        System.out.println("Dengue ML web app: http://localhost:" + port);
        System.out.println("Also try: http://127.0.0.1:" + port);
    }

    private static void sendJson(io.javalin.http.Context ctx, int status, Object body) {
        ctx.status(status).contentType("application/json").result(GSON.toJson(body));
    }

    private static String loadIndexHtml() {
        try (InputStream in = WebApp.class.getResourceAsStream("/public/index.html")) {
            if (in == null) throw new IllegalStateException("Missing /public/index.html on classpath");
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (Exception e) {
            throw new RuntimeException("Could not load index.html", e);
        }
    }

    private static int getInt(Map<String, Number> m, String key, int def) {
        if (!m.containsKey(key)) return def;
        return m.get(key).intValue();
    }

    private static Double[] getSampleDengueCases() {
        double[] a = sampleDengueCases();
        Double[] out = new Double[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i];
        return out;
    }

    private static double[] sampleDengueCases() {
        return new double[] {
            45, 52, 61, 78, 88, 95, 102, 98, 85, 72, 58, 48,
            50, 55, 65, 82, 92, 100, 108, 104, 88, 75, 62, 51,
            48, 54, 68, 85, 94, 103, 112, 106, 90, 78, 64, 52,
            52, 58, 70, 86, 96, 105, 115, 108, 92, 80, 66, 55
        };
    }
}
