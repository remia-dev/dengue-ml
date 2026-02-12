# Dengue ML Predictor

Java project for **statistical modeling** (linear regression) and **predictive modeling** (Seasonal ARIMA – SARIMA) applied to dengue case data.

## Models

### 1. Linear regression (statistical modeling)

- **Formula**: \( y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n \)
- **Estimation**: Ordinary Least Squares via the **normal equation**:  
  \( \beta = (X'X)^{-1} X'y \),  
  where \( X \) is the design matrix (with a column of 1s for the intercept) and \( y \) is the response (e.g. dengue cases).
- Use for explaining relationships (e.g. cases vs time, rainfall, season) and R² / adjusted R².

### 2. SARIMA (predictive modeling)

- **Model**: Seasonal Autoregressive Integrated Moving Average  
  SARIMA\((p,d,q)(P,D,Q)_s\)
  - \( p, d, q \): non-seasonal AR order, differencing, MA order  
  - \( P, D, Q \): seasonal AR, seasonal differencing, seasonal MA  
  - \( s \): season length (e.g. 12 for monthly data with yearly seasonality)
- Differencing: \( \nabla^d \nabla_s^D y_t \) (regular and seasonal).
- Used to **forecast** future dengue cases from historical counts.

## Build and run

**Requirements:** Java 17+, Maven 3.6+

```bash
cd proj
mvn compile
mvn exec:java -Dexec.mainClass="dengue.Main"
```

Or with a CSV file (first column = cases, optional extra columns = covariates):

```bash
mvn exec:java -Dexec.mainClass="dengue.Main" -Dexec.args="src/main/resources/sample_dengue_data.csv"
```

To create an executable JAR:

```bash
mvn package
java -cp target/dengue-predictor-1.0.0.jar dengue.Main
```

(Note: the JAR depends on `commons-math3`; use the same classpath or a fat JAR if you move it.)

### Web app

Browser-based UI to run the same models: paste or load dengue case counts, tune SARIMA parameters, and view linear regression stats plus SARIMA forecasts.

**Run the server:**

```bash
mvn exec:java -Pweb
```

Or: `mvn exec:java -Dexec.mainClass="dengue.WebApp"`

Then open **http://localhost:7000** (or **http://127.0.0.1:7000**) in your browser. The server binds to `0.0.0.0:7000`, so it is reachable from the host when running in WSL or in a container. After changing code or the UI: stop the server, run `mvn compile`, start again, and hard-refresh the page (Ctrl+Shift+R) to avoid cached files.

**What you can do in the UI:**

- **Case counts** — Paste numbers in the text area (comma-, space-, or newline-separated), or click **Use sample data** to load 48 months of example dengue cases.
- **SARIMA parameters** — Optionally change p, d, q, P, D, Q, season length *s*, and **Forecast steps** (1–60). Default is SARIMA(1,0,1)(1,0,1)12 with 6 forecast steps.
- **Analyze** — Runs linear regression (cases vs time) and SARIMA, then shows intercept β₀, R², adjusted R², and the next N periods’ forecast.

**Data:** Provide at least 24 values (e.g. 2 years of monthly data) for reasonable results.

**Deploy on Railway:** The app reads `PORT` from the environment and builds a single runnable JAR with the `web` profile. From the project root:

1. Install the [Railway CLI](https://docs.railway.app/develop/cli) (optional) or use the [Railway dashboard](https://railway.app).
2. Create a new project and connect your repo (or run `railway init` in the repo).
3. Add a new service; choose this repo. Railway will use `nixpacks.toml` to run:
   - **Build:** `mvn clean package -Pweb -DskipTests` → produces `target/dengue-predictor-web.jar`
   - **Start:** `java -jar target/dengue-predictor-web.jar`
4. Deploy. The service will get a public URL; open it to use the web app.

No extra env vars are required; Railway sets `PORT` automatically.

**REST API:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI (HTML). |
| GET | `/api/health` | Health check; returns `{"status":"ok","port":7000}`. |
| GET | `/api/sample` | Sample dengue case series (JSON array). |
| POST | `/api/analyze` | Run both models. Body: `{"cases": [...], "sarima": {"p", "d", "q", "P", "D", "Q", "s"}, "forecastSteps": 6}`. Returns `{"regression": {...}, "forecast": [...]}` or `{"error": "..."}`. |

Example: `POST /api/analyze` with body `{"cases": [45, 52, 61, 78, 88, 95, 102, 98, 85, 72, 58, 48], "forecastSteps": 6}`.


## Project layout

- `src/main/java/dengue/ml/LinearRegression.java` – OLS linear regression \((X'X)^{-1}X'y\)
- `src/main/java/dengue/ml/Sarima.java` – SARIMA fitting and forecasting
- `src/main/java/dengue/ml/DenguePredictor.java` – Combines regression + SARIMA, CSV loading
- `src/main/java/dengue/Main.java` – CLI demo
- `src/main/java/dengue/WebApp.java` – Web server (Javalin) and REST API
- `src/main/resources/public/index.html` – Web UI
- `src/main/resources/sample_dengue_data.csv` – Example dengue data

## Data

- Use **monthly (or weekly) dengue case counts** for SARIMA; set \( s = 12 \) for monthly yearly seasonality.
- For linear regression, provide covariates (e.g. time index, month, rainfall) as extra columns in the CSV or via the API.

## License

Use and modify as needed for your project.
