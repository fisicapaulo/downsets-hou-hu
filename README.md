# downsets-hou-hu

Material suplementar do artigo **Formulações Inteiras e Saturação Complementar na Redução por Downsets de Hou–Hu**, de Paulo Vieira.

Este repositório reúne os scripts de enumeração e validação computacional do artigo. Os três arquivos ficam na raiz, e os testes padrão não exigem catálogos ou relatórios JSON preexistentes.

## Scripts e resultados esperados

| Script | Função | Resultado esperado |
| --- | --- | --- |
| `enumerar_r7.py` | Enumera as famílias disponíveis, saturadas e *shifted* em sete coordenadas. | 134 famílias. |
| `validar_r456.py` | Compara as fórmulas de Hou–Hu com os modelos MILP original e reduzido. | 2.244 instâncias sem divergências de valor: 495, 1.287 e 462 para quatro, cinco e seis partes, respectivamente. |
| `validar_r7.py` | Compara os dois MILPs, o máximo no catálogo *shifted* e o máximo entre as 43 famílias canônicas. | 122 instâncias da grade autônoma sem divergências de valor. |

## Como reproduzir

Na raiz do repositório, instale as dependências dos validadores e execute os scripts:

```bash
python -m pip install numpy "scipy>=1.9"
python enumerar_r7.py --output resultados/shifted_candidates_r7.json
python validar_r456.py
python validar_r7.py
```

O enumerador usa apenas a biblioteca padrão do Python. Os validadores importam funções de `enumerar_r7.py`, mas não precisam que ele tenha sido executado antes. O validador de sete partes regenera o catálogo em memória.

No **Google Colab**, coloque os três arquivos na mesma pasta e execute os comandos a partir dela, com `!` no início de cada linha. Para um teste rápido do validador de quatro a seis partes, use `python validar_r456.py --quick`.

Os validadores gravam os relatórios em `resultados/validation_r456.json` e `resultados/validation_r7.json`, sem download automático.

## Grade de sete partes

A grade autônoma contém todos os 120 vetores não crescentes de sete entradas entre 2 e 5, além de `(10, 2, 2, 2, 2, 2, 2)` e `(10, 10, 10, 10, 10, 10, 10)`.

Para repetir a validação com os vetores de um relatório histórico, forneça o arquivo explicitamente:

```bash
python validar_r7.py --historical caminho/para/validation_r7.json
```

Esse modo lê apenas os vetores `n`, recalcula os valores e grava `resultados/validation_r7_historical.json`. A grade autônoma não pressupõe que seus 122 vetores coincidam com os da rodada histórica.

## Alcance da validação

A viabilidade das soluções retornadas pelos MILPs e seus pesos são verificados com inteiros Python. A otimalidade dos MILPs é reportada numericamente pelo SciPy/HiGHS, sem certificado formal exato do limite dual.

A comparação é de **valores objetivos**: famílias diferentes podem atingir o mesmo valor. A concordância na grade testada é evidência computacional, não uma prova da conjectura para sete partes.
