"""Builder for notebooks/elastic_constants.ipynb (valid JSON via nbformat).

Run: .venv/bin/python notebooks/_build_elastic_notebook.py
Then execute: .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
              --ExecutePreprocessor.timeout=2400 notebooks/elastic_constants.ipynb
"""

import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        "# Elastic constants with `pyiron_workflow_atomistics`\n"
        "\n"
        "This notebook computes the full elastic stiffness tensor and every derived constant in the\n"
        "[Materials Project elasticity methodology](https://docs.materialsproject.org/methodology/materials-methodology/elasticity)\n"
        "using `pyiron_workflow_atomistics.physics.elastic`, driven by **any ASE calculator**.\n"
        "\n"
        "**Method (same as Materials Project, via `pymatgen.analysis.elasticity`):**\n"
        "1. Relax the structure (cell + ions).\n"
        "2. Apply the MP-standard deformation set (normal strains ±0.5%/±1%, shear ±3%/±6% → 24 cells).\n"
        "3. Relax ions at **fixed cell** for each deformation and read the stress.\n"
        "4. Fit the 6×6 stiffness tensor `C_ij`; derive K, G (Voigt/Reuss/Hill), Young's E, Poisson ν, "
        "universal anisotropy A_U, and a Born mechanical-stability check.\n"
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 1. Fast example with the EMT calculator (Cu)\n"
        "EMT is cheap and ships with ASE, so this cell runs end-to-end in ~20 s."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "from ase.build import bulk\n"
        "from ase.calculators.emt import EMT\n"
        "from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic\n"
        "from pyiron_workflow_atomistics.physics.elastic import calculate_elastic_constants\n"
        "\n"
        "structure = bulk('Cu', 'fcc', a=3.615, cubic=True)\n"
        "engine = ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(),\n"
        "                   working_directory='elastic_emt_cu')\n"
        "wf = calculate_elastic_constants(structure=structure, engine=engine, relax_initial=True)\n"
        "out = wf.run()\n"
        "d = out['elastic_constants']\n"
        "print('mechanically stable:', d['mechanically_stable'])"
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "### The full 6×6 stiffness tensor (GPa, IEEE orientation)"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "import numpy as np, pandas as pd\n"
        "C = np.array(d['elastic_tensor_ieee'])\n"
        "pd.DataFrame(np.round(C, 1),\n"
        "             index=[f'{i+1}' for i in range(6)],\n"
        "             columns=[f'{j+1}' for j in range(6)])"
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "### Derived elastic constants (the full Materials-Project list)"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "summary = {\n"
        "    'K_Voigt': d['K_Voigt'], 'K_Reuss': d['K_Reuss'], 'K_VRH': d['K_VRH'],\n"
        "    'G_Voigt': d['G_Voigt'], 'G_Reuss': d['G_Reuss'], 'G_VRH': d['G_VRH'],\n"
        "    \"Young's E\": d['youngs_modulus'], 'Poisson nu': d['poisson_ratio'],\n"
        "    'Universal anisotropy A_U': d['universal_anisotropy'],\n"
        "    'Mechanically stable': d['mechanically_stable'],\n"
        "}\n"
        "pd.Series(summary).to_frame('value (GPa where applicable)')"
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 2. Swapping in the GRACE-2L-SMAX foundation model\n"
        "The only change is the ASE calculator. The SMAX models are cached at `/ptmp/hmai/grace_cache`\n"
        "(set `GRACE_CACHE`). This cell is guarded so the notebook still runs where GRACE/TensorFlow\n"
        "isn't installed; set `RUN_GRACE=1` (and ideally run on a GPU node) to execute it live."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "import os\n"
        "os.environ.setdefault('GRACE_CACHE', '/ptmp/hmai/grace_cache')\n"
        "os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')\n"
        "os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')\n"
        "\n"
        "if os.environ.get('RUN_GRACE'):\n"
        "    import sys\n"
        "    sys.path.insert(0, os.path.abspath('../verification'))\n"
        "    from elastic_grace_vs_mp import load_grace_calculator\n"
        "    gengine = ASEEngine(EngineInput=CalcInputStatic(),\n"
        "                        calculator=load_grace_calculator('GRACE-2L-SMAX-large'),\n"
        "                        working_directory='elastic_grace_cu')\n"
        "    gout = calculate_elastic_constants(structure=bulk('Cu','fcc',a=3.615,cubic=True),\n"
        "                                       engine=gengine, relax_initial=True).run()\n"
        "    gd = gout['elastic_constants']\n"
        "    print('GRACE-2L-SMAX-large Cu:  K_VRH=%.1f  G_VRH=%.1f  C11=%.1f  C12=%.1f  C44=%.1f'\n"
        "          % (gd['K_VRH'], gd['G_VRH'], gd['elastic_tensor_ieee'][0][0],\n"
        "             gd['elastic_tensor_ieee'][0][1], gd['elastic_tensor_ieee'][3][3]))\n"
        "else:\n"
        "    print('Set RUN_GRACE=1 to run the GRACE-2L-SMAX-large calculator here.')"
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 3. GRACE-2L-SMAX-large vs Materials Project DFT (6 elements)\n"
        "The script `verification/elastic_grace_vs_mp.py` runs the workflow with GRACE-2L-SMAX-large\n"
        "for Al, Cu, Si, Fe, Ni, W and cross-checks the Materials Project elastic tensors. The table\n"
        "below is loaded from the committed results if present."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "import os, pandas as pd\n"
        "csv_path = os.path.join('..', 'verification', 'results', 'combined.csv')\n"
        "if os.path.exists(csv_path):\n"
        "    cols = ['material','mp_id','mp_state','grace_K','mp_K','exp_K',\n"
        "            'grace_G','mp_G','exp_G','grace_C11','mp_C11','grace_C44','mp_C44',\n"
        "            'ref_used','K_err_vs_best_%','G_err_vs_best_%','grace_stable']\n"
        "    df = pd.read_csv(csv_path)\n"
        "    display(df[[c for c in cols if c in df.columns]])\n"
        "else:\n"
        "    print('Run verification/submit_gpudev.sh + aggregate.py to populate this table.')"
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "**Notes.** Materials Project's own elastic fit for Al (mp-134) is flagged `failed` "
        "(mechanically unstable), so the comparison uses experimental single-crystal constants there. "
        "Fe and Ni are ferromagnetic — GRACE-SMAX handles the magnetism internally. See "
        "`verification/SUMMARY.md` for the full table and interpretation."
    )
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out_path = os.path.join(os.path.dirname(__file__), "elastic_constants.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print("wrote", out_path)
