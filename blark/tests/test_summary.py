from dataclasses import dataclass
from typing import Optional

import pytest

from .. import transform as tf
from ..dependency_store import DependencyStore, PlcProjectMetadata
from ..summary import DeclarationSummary


@dataclass
class DeclarationCheck:
    name: str
    base_type: str
    value: Optional[str] = None
    location: Optional[str] = None
    comments: Optional[list[str]] = None


def check_declarations(
    container: dict[str, DeclarationSummary],
    checks: list[DeclarationCheck]
):
    """
    Check that ``container`` has the listed declarations - and only those.

    Parameters
    ----------
    container : dict[str, DeclarationSummary]
    checks : list[DeclarationCheck]
    """
    for check in checks:
        try:
            summary = container[check.name]
        except KeyError:
            raise KeyError(f"{check.name} not listed in the declaration section")

        assert summary.base_type == check.base_type
        if check.value is None:
            assert summary.value == "None"  # TODO?
            assert summary.item.init.value is None
        else:
            assert summary.value == check.value
            assert str(summary.item.init.value) == check.value

        if check.comments is not None:
            comments = list(str(comment) for comment in summary.item.meta.comments)
            assert comments == check.comments

        if check.location is not None:
            assert summary.location == check.location

    all_decls = {str(decl) for decl in container}
    expected_decls = {str(decl.name) for decl in checks}
    assert all_decls == expected_decls


@pytest.fixture
def project_a(store: DependencyStore) -> PlcProjectMetadata:
    # This is project a from the git submodule included with the blark test
    # suite
    proj, = store.get_dependency("project_a")
    return proj


def test_project_a_summary(project_a: PlcProjectMetadata):
    summary = project_a.summary

    assert list(summary.function_blocks) == ["FB_ProjectA"]

    fb_projecta = summary.function_blocks["FB_ProjectA"]
    assert list(fb_projecta.declarations) == ["fValue"]
    assert fb_projecta.declarations["fValue"].base_type == "LREAL"
    assert "fValue := 1.0;" in str(fb_projecta.implementation)

    st_projecta = summary.data_types["ST_ProjectA"]
    assert list(st_projecta.declarations) == ["iValue"]
    assert st_projecta.declarations["iValue"].base_type == "INT"

    main = summary.programs["MAIN"]
    assert main.implementation is None
    assert list(main.declarations) == []

    gvl = summary.globals["GVL_ProjectA"]
    gvl_bprojecta = gvl.declarations["bProjectA"]
    assert gvl_bprojecta.base_type == "BOOL"
    assert isinstance(gvl_bprojecta.item.init, tf.TypeInitialization)
    assert str(gvl_bprojecta.item.init.value) == "FALSE"


@pytest.fixture
def project_b(store: DependencyStore) -> PlcProjectMetadata:
    # This is project b from the git submodule included with the blark test
    # suite
    proj, = store.get_dependency("project_b")
    return proj


def test_project_b_summary(project_b: PlcProjectMetadata):
    summary = project_b.summary

    st_projectb = summary.data_types["ST_ProjectB"]
    assert list(st_projectb.declarations) == ["iProjectB"]
    assert st_projectb.declarations["iProjectB"].base_type == "INT"

    main = summary.programs["MAIN"]
    assert str(main.implementation) == "fbLogger();"
    assert list(main.declarations) == ["bProjectB", "fbLogger"]

    gvl = summary.globals["GVL_ProjectB"]
    gvl_bprojecta = gvl.declarations["bProjectB"]
    assert gvl_bprojecta.base_type == "BOOL"
    assert isinstance(gvl_bprojecta.item.init, tf.TypeInitialization)
    assert gvl_bprojecta.item.init.value is None


@pytest.fixture
def twincat_general_281(store: DependencyStore) -> PlcProjectMetadata:
    # This is project lcls-twincat-general v2.8.1 from the git submodule
    # included with the blark test suite.
    # This is *not* the full version from pcdshub/lcls-twincat-general, but
    # rather a part of it for the purposes of the blark test suite.
    proj, = store.get_dependency("LCLS_General", "v2.8.1")
    return proj


def test_twincat_general(twincat_general_281: PlcProjectMetadata):
    summary = twincat_general_281.summary
    assert list(summary.function_blocks) == ["FB_LogHandler"]
    loghandler = summary.function_blocks["FB_LogHandler"]

    check_declarations(
        loghandler.declarations,
        [
            DeclarationCheck(
                name="bLogHandlerInput",
                base_type="BOOL",
                value=None,
            ),
            DeclarationCheck(
                name="nNumListeners",
                base_type="UINT",
                value="6",
            ),
            DeclarationCheck(
                name="DisarmCountDefault",
                base_type="UINT",
                value="5",
            ),
        ]
    )
    # Ensure comments transferred over
    assert loghandler.implementation is None

    impl = str(loghandler["CircuitBreaker"].implementation)
    assert "Logic explanation" in impl
    assert "// reset the count for the next" in impl
    # Check about reformatting
    assert "bTripCon := GVL_Logger.nGlobAccEvents > 0;" in impl

    assert list(summary.globals) == ["DefaultGlobals", "Global_Variables_EtherCAT"]
    ecat = summary.globals["Global_Variables_EtherCAT"]

    check_declarations(
        ecat.declarations,
        [
            DeclarationCheck(
                name="iSLAVEADDR_ARR_SIZE",
                base_type="UINT",
                value="256",
            ),
            DeclarationCheck(
                name="ESC_MAX_PORTS",
                base_type="UINT",
                value="3",
                comments=["// Maximum number of ports (4) on ESC"],
            ),
        ]
    )

    defaults = summary.globals["DefaultGlobals"]

    check_declarations(
        defaults.declarations,
        [
            DeclarationCheck(
                name="stSys",
                base_type="ST_System",
                comments=["//Included for you"],
            ),
            DeclarationCheck(
                name="fTimeStamp",
                base_type="LREAL",
            ),
        ]
    )

    system = summary.data_types["ST_System"]

    check_declarations(
        system.declarations,
        [
            DeclarationCheck(
                name="xSwAlmRst",
                base_type="BOOL",
                comments=["(* Global Alarm Reset - EPICS Command *)"],
            ),
            DeclarationCheck(
                name="xAtVacuum",
                base_type="BOOL",
                comments=["(* System At Vacuum *)"],
            ),
            DeclarationCheck(
                name="xFirstScan",
                base_type="BOOL",
                comments=[
                    "(* This boolean is true for the first scan, and is false "
                    "thereafter, use for initialization of stuff *)",
                ],
            ),
            DeclarationCheck(
                name="xOverrideMode",
                base_type="BOOL",
                comments=[
                    "//This bit is set when using the override features of the system"
                ],
            ),
            DeclarationCheck(
                name="xIOState",
                base_type="BOOL",
                comments=["(* ECat Bus Health *)"],
            ),
            DeclarationCheck(
                name="I_EcatMaster1",
                base_type="AMSNETID",
                location="input",
                comments=[
                    "{attribute 'naming' := 'omit'}",
                    "(* AMS Net ID used for FB_EcatDiag, among others *)"
                ],
            ),
        ],
    )
