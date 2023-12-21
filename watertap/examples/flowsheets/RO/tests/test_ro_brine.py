#################################################################################
# adapted from test_nf.py
#################################################################################

import pytest
from pyomo.environ import value
from watertap.examples.flowsheets.RO.ro_brine import main


@pytest.mark.requires_idaes_solver
@pytest.mark.component
def test_main():
    m = main()
    test_dict = {
        "area": [m.fs.unit.area, 1126.7194556],
        "ro_recovery": [
            m.fs.unit.recovery_vol_phase[0.0, "Liq"] * 100,
           77.004208,
        ],
        "licl_recovery": [
            m.fs.unit.recovery_mass_phase_comp[0,"Liq","LiCl"].value*100,
            7.519423,
        ]
    }
    for model_result, testval in test_dict.values():
        assert pytest.approx(testval, rel=1e-3) == value(model_result)