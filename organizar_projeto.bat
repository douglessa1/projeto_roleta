@echo off
echo ================================
echo   Limpando e organizando projeto
echo ================================

REM Criar pasta static se nao existir
if not exist static (
    mkdir static
)

REM Mover arquivos do frontend para /static
move /Y index.html static\ >nul 2>&1
move /Y style.css static\ >nul 2>&1
move /Y app.js static\ >nul 2>&1
move /Y main.js static\ >nul 2>&1

REM Pastas que devem permanecer
set KEEP_DIRS=ml_models venv static __pycache__

REM Apagar arquivos desnecessarios
for %%F in (
    analise_projeto_roleta.md
    betting_systems.html
    documentation.html
    dashboard-logic.js
    exemplo-backend.js
    exemplo-integracao.js
    integracao.html
    login.html
    ml_analysis.html
    passo-a-passo.html
    README-INTEGRACAO.md
    relatorio_melhorias.md
    suporte.html
) do (
    if exist %%F (
        del /F /Q %%F
        echo Apagado: %%F
    )
)

REM Apagar pastas inutilizadas
for %%D in (
    backend
    resources
    supabase
    data
    .qodo
    .vscode
    .history
) do (
    if exist %%D (
        echo Excluindo pasta: %%D
        rmdir /S /Q %%D
    )
)

echo ================================
echo    Projeto organizado com sucesso!
echo ================================
pause
