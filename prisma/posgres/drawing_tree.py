drawing_tree = {
    "Main&Transfer":
    {
        "115_tie": {
            "mandatory": {
                "115_breaker": 1,
                "ds": {
                    "_total": 2,
                    "115_3way_ds_w_motor": 1,
                    "115_1way_ds_w_motor": 1,
                    "115_ds": 1,
                },
                "LL": 1,
                "v_m_digital": 1,
                "vt": {
                    "_total": 1,
                    "115_cvt_1p": 1,
                    "115_cvt_3p": 1,
                    "115_vt_1p": 1,
                    "115_vt_3p": 1,
                }
            },
        },
        "115_transformer": {
            "mandatory": {
                "115_breaker": 1,
                "ds": {
                    "_total": 3,
                    "115_3way_ds_w_motor": 1,
                    "115_1way_ds_w_motor": 1,
                    "115_ds": 1,
                },
                "115_la": 1,
                "tx": {
                    "_total": 1,
                    "11522_tx_dyn1": 1,
                    "11522_tx_ynyn0d1": 1,
                },
                "ngr": {
                    "_total": 1,
                    "NGR": 1,
                    "NGR_future": 1,
                }, },
            "optional": {
                "DIM": 1,
                "DPM": 1, }
        },
        "115_incoming": {
            "mandatory": {
                "115_breaker": 1,
                "ds": {
                    "_total": 3,
                    "115_3way_ds_w_motor": 1,
                    "115_1way_ds_w_motor": 1,
                    "115_ds": 1,
                },
                "115_gs": 1,
                "115_la": 1,
                "ss_man_mode": 1,
                "DPM": 1,
                "LL": 1,
                "vt": {
                    "_total": 1,
                    "115_cvt_1p": 1,
                    "115_cvt_3p": 1,
                    "115_vt_1p": 1,
                    "115_vt_3p": 1,
                }},
            "optional": {
                "DIM": 1,
            }
        },
        "22_tie": {
            "mandatory": {
                "22_breaker": 1,
                "vt": {
                    "_total": 2,
                    "22_vt_1p": 1,
                    "22_vt_3p": 1,
                    "22_cvt_1p": 1,
                    "22_cvt_3p": 1,
                },
                "DPM": 1,
                "LL": 2,
            },
        },
        "22_capacitor": {
            "mandatory": {
                "22_breaker": 1,
                "22_cap_bank": 1,
                "22_gs": 1,
                "22_ll": 1,
                "Q": 1,
                "terminator": {
                    "_total": 2,
                    "terminator_w_future": 1,
                    "terminator_single": 1,
                },
            },
            "optional": {
                "DIM": 1,
                "DPM": 1,
            }
        },
        "22_outgoing": {
            "mandatory": {
                "22_breaker": 1,
                "out": {
                    "_total": 1,
                    "22_ds_la_out": 1,
                    "22_ds_out": 1,
                },
                "22_gs": 1,
                "22_ll": 1,
                "terminator": {
                    "_total": 2,
                    "terminator_w_future": 1,
                    "terminator_single": 1,
                    "terminator_double": 1,
                },
            },
            "optional": {
                "DIM": 1,
                "DPM": 1,
            }
        },
        "22_service": {
            "mandatory": {
                "22_ds": 1,
                "22_ll": 1,
                "22_gs": 1,
                "terminator": {
                    "_total": 2,
                    "terminator_w_future": 1,
                    "terminator_single": 1,
                    "terminator_double": 1,
                },
            },
        },
        "22_incoming": {
            "mandatory": {
                "22_breaker": 1,
                "22_gs": 1,
                "22_ll": 1,
                "LL": 1,
                "v_m": 1,
                "terminator": {
                    "_total": 2,
                    "terminator_single": 1,
                    "terminator_double": 1,
                },
            },
            "optional": {
                "DIM": 1,
                "DPM": 1,
            }
        }
    }}
