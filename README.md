# Workflow-CI — MLflow CI Pipeline

Repositori ini berisi workflow CI menggunakan GitHub Actions untuk melatih model ML secara otomatis dengan MLflow Projects.

## Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                                  # GitHub Actions CI
└── MLProject/
    ├── MLProject                                   # Konfigurasi MLflow Project
    ├── conda.yaml                                  # Environment dependencies
    ├── modelling.py                                # Script training
    └── loan_approval_dataset_preprocessing.csv    # Dataset siap latih
```

## Cara Kerja CI

Workflow otomatis terpantik saat:
- Push ke branch `main`
- Trigger manual (`workflow_dispatch`)

### Tahapan CI
1. ✅ Checkout repository
2. ✅ Setup Python 3.10
3. ✅ Install dependencies
4. ✅ Run `mlflow run MLProject/`
5. ✅ Upload artefak `mlruns/` ke GitHub Actions Artifacts

## Jalankan Lokal

```bash
# Masuk ke folder MLProject
cd MLProject

# Jalankan dengan mlflow
mlflow run . --env-manager=local \
    -P data_path=loan_approval_dataset_preprocessing.csv \
    -P n_estimators=100 \
    -P max_depth=10
```
