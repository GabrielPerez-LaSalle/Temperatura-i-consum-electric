## Projecte integrador — Temperatura i consum elèctric

### 🌍 Context

El consum d’energia elèctrica en una ciutat està fortament influït per les condicions meteorològiques. Quan fa molt fred, la demanda d’electricitat augmenta per la calefacció; quan fa molta calor, també pot augmentar per l’ús d’aire condicionat. Aquest comportament és un exemple perfecte per estudiar com la temperatura pot influir en el consum energètic i per posar en pràctica els conceptes de regressió lineal i avaluació de models.

En aquest projecte treballarem amb el dataset de **Kaggle**:

> **Hourly energy demand, generation, prices and weather – Spain**
> [https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)

Aquest conjunt de dades recull informació horària sobre **consum elèctric**, **temperatura** i altres variables a Espanya durant diversos anys. Per a aquesta pràctica, simplificarem les dades per treballar **a nivell diari**, mantenint només les columnes necessàries.

---

### 🌟 Objectius d’aprenentatge

1. Comprendre la relació entre una variable independent (temperatura) i una variable dependent (consum elèctric).
2. Construir i interpretar un model de regressió lineal utilitzant *scikit-learn*.
3. Aplicar mètriques d’avaluació com el MSE i el R².
4. Visualitzar els resultats i interpretar la tendència del consum segons la temperatura.
5. Reflexionar sobre la idoneïtat del model lineal i introduir la idea de regressió polinòmica.

---

### 📊 Dataset i preparació

El dataset original conté dades **horàries**, però per a la regressió lineal simple ens interessa **agregar-les per dia**. Les columnes bàsiques que necessitem són:

* **Date**: data del registre
* **Temperature**: temperatura mitjana diària (°C)
* **EnergyConsumption**: consum elèctric total o mitjà diari (MWh)

#### 🧰 Pistes per preparar les dades

1. Descarrega els datasets i carrega’l amb Pandas. Uneix els dos en un sol DataFrame

2. Assegura’t que la columna de data estigui en format datetime

3. Extreu la temperatura i el consum d’energia (per exemple, demanda elèctrica nacional)

4. Agrega per dia, calculant la mitjana de la temperatura i la suma del consum

5. Ja tens les dades diàries preparades per analitzar!

---

### 🧮 Tasques a realitzar

#### 1️⃣ Exploració inicial de les dades

* Carrega el DataFrame i comprova quantes files i columnes té.
* Mostra les primeres 5 files.
* Fes un gràfic de dispersió de temperatura vs consum.

#### 2️⃣ Entrenament del model de regressió lineal

* Crea les variables `X` (Temperature) i `y` (EnergyConsumption).
* Entrena un model de regressió lineal.
* Mostra la pendent (`w`) i l’intercept (`b`).
* Interpreta els valors obtinguts.

#### 3️⃣ Avaluació del model

* Calcula el MSE i el R² del model.
* Interpreta el significat de cada mètrica.
* Reflexiona si el model explica prou bé la variabilitat del consum.

#### 4️⃣ Visualització dels resultats

* Representa la recta ajustada sobre el gràfic de dispersió.
* Comenta si el model s’ajusta bé a les dades o si detectes una relació corba.

#### 5️⃣ Anàlisi amb un subset

* Veient que aquest model no s'ajusta gaire, provem de retallar les dades aviam si observem un comportament lineal
* Comprova si ara el model s'ajusta millor


####  EXTRA — regressió polinòmica

* Si observes una relació corba, prova un model polinòmic de grau 2.
* Compara el R² del model lineal i del model polinòmic.

---

### 🧠 Preguntes de reflexió

1. Quin signe té la pendent (w)? Què ens diu sobre la relació entre temperatura i consum?
2. Quin seria el consum previst si la temperatura fos 0 °C? És raonable?
3. El model lineal explica bé el comportament real de les dades?
4. Què podries fer per millorar el model?
5. Quins altres factors poden afectar el consum elèctric (a part de la temperatura)?
6. En quins contextos seria inadequat aplicar directament aquest model?
7. Quines implicacions ètiques o de privacitat podria tenir analitzar dades de consum?

---

### 📘 Conclusions

* Escriu aquí les conclusions que pots extreure