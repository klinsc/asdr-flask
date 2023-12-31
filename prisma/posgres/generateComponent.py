import asyncio
from prisma import Prisma


async def main() -> None:
    prisma = Prisma()
    await prisma.connect()

    # write your queries here
    await prisma.component.create_many(
        skip_duplicates=True,
        data=[
            {
                "index": 0,
                "name": "11522_tx_dyn1",
                "color": "#6344B2",
                "part": "kv115"
            },
            {
                "index": 1,
                "name": "11522_tx_ynyn0d1",
                "color": "#3C57B0",
                "part": "kv115"
            },
            {
                "index":  2,
                "name": "115_1way_ds_w_motor",
                "color": "#B2AA0A",
                "part": "kv115"
            },
            {
                "index": 3,
                "name": "115_3ways_ds",
                "color": "#B20D0D",
                "part": "kv115"
            },
            {
                "index": 4,
                "name": "115_3ways_ds_w_motor",
                "color": "#B20D0D",
                "part": "kv115"
            },
            {
                "index": 5,
                "name": "115_breaker",
                "color": "#F20D7A",
                "part": "kv115"
            },
            {
                "index": 6,
                "name": "115_buffer",
                "color": "#B20D0D",
                "part": "kv115"
            },
            {
                "index":    7,
                "name": "115_cvt_1p",
                "color": "#1CBCD2",
                "part": "kv115"
            },
            {
                "index": 8,
                "name": "115_cvt_3p",
                "color": "#E62A64",
                "part": "kv115"
            },
            {
                "index": 9,
                "name": "115_ds",
                "color": "#B20D0D",
                "part": "kv115"
            },
            {
                "index": 10,
                "name": "115_gs",
                "color": "#B20D3C",
                "part": "kv115"
            },
            {
                "index": 11,
                "name": "115_gs_w_motor",
                "color": "#f287f5",
                "part": "kv115"
            },
            {
                "index": 12,
                "name": "115_la",
                "color": "#9936AC",
                "part": "kv115"
            },
            {
                "index": 13,
                "name": "115_vt_1p",
                "color": "#F1463F",
                "part": "kv115"
            },
            {
                "index": 14,
                "name": "115_vt_3p",
                "color": "#199588",
                "part": "kv115"
            },
            {
                "index": 15,
                "name": "22_breaker",
                "color": "#3C0DB2",
                "part": "kv22"
            },
            {
                "index": 16,
                "name": "22_cap_bank",
                "color": "#0DB24C",
                "part": "kv22"
            },
            {
                "index": 17,
                "name": "22_ds",
                "color": "#7AB20D",
                "part": "kv22"
            },
            {
                "index": 18,
                "name": "22_ds_out",
                "color": "#B2AA0A",
                "part": "kv22"
            },
            {
                "index": 19,
                "name": "22_ds_la_out",
                "color": "#0DB20D",
                "part": "kv22"
            },
            {
                "index": 20,
                "name": "22_gs",
                "color": "#0D0DB2",
                "part": "kv22"
            },
            {
                "index": 21,
                "name": "22_ll",
                "color": "#16448c",
                "part": "kv22"
            },
            {
                "index": 22,
                "name": "22_vt_1p",
                "color": "#3CB20D",
                "part": "kv22"
            },
            {
                "index": 23,
                "name": "22_vt_3p",
                "color": "#1F9AEE",
                "part": "kv22"
            },
            {
                "index": 24,
                "name": "BCU",
                "color": "#B88000",
                "part": "universal"
            },
            {
                "index": 25,
                "name": "DIM",
                "color": "#B20D0D",
                "part": "universal"
            },
            {
                "index": 26,
                "name": "DPM",
                "color": "#B24CCC",
                "part": "universal"
            },
            {
                "index": 27,
                "name": "LL",
                "color": "#0DB2B2",
                "part": "universal"
            },
            {
                "index": 28,
                "name": "MU",
                "color": "#0DACEF",
                "part": "universal"
            },
            {
                "index": 29,
                "name": "NGR",
                "color": "#B20DFF",
                "part": "kv22"
            },
            {
                "index": 30,
                "name": "NGR_future",
                "color": "#B20DD0",
                "part": "kv22"
            },
            {
                "index": 31,
                "name": "Q",
                "color": "#7A0DB2",
                "part": "kv22"
            },
            {
                "index": 32,
                "name": "remote_io_module",
                "color": "#0DB2B2",
                "part": "kv115"
            },
            {
                "index": 33,
                "name": "ss_man_mode",
                "color": "#F20D3C",
                "part": "kv115"
            },
            {
                "index": 34,
                "name": "tele_protection",
                "color": "#0DACEF",
                "part": "kv115"
            },
            {
                "index": 35,
                "name": "terminator_double",
                "color": "#0DB27A",
                "part": "kv22"
            },
            {
                "index": 36,
                "name": "terminator_single",
                "color": "#0DB2B2",
                "part": "kv22"
            },
            {
                "index": 37,
                "name": "terminator_splicing_kits",
                "color": "#FD7AB2",
                "part": "kv22"
            },
            {
                "index": 38,
                "name": "terminator_w_future",
                "color": "#0D7AB2",
                "part": "kv22"
            },
            {
                "index": 39,
                "name": "v_m",
                "color": "#0DB24C",
                "part": "kv22"
            },
            {
                "index": 40,
                "name": "v_m_digital",
                "color": "#0DB27A",
                "part": "universal"
            },
        ]
    )

    await prisma.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
