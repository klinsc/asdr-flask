import asyncio
from prisma import Prisma

# <View>
#     <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true" />
#     <RectangleLabels name="label" toName="image">
#         <Label value="11522_tx_dyn1" background="#6344B2" />
#         <Label value="11522_tx_ynyn0d1" background="#3C57B0" />
#         <Label value="115_1way_ds_w_motor" background="#B2AA0A" />
#         <Label value="115_3ways_ds" background="#B20D0D" />
#         <Label value="115_3ways_ds_w_motor" background="#B20D0D" />
#         <Label value="115_breaker" background="#F20D7A" />
#         <Label value="115_buffer" background="#B20D0D" />
#         <Label value="115_cvt_1p" background="#1CBCD2" />
#         <Label value="115_cvt_3p" background="#E62A64" />
#         <Label value="115_ds" background="#B20D0D" />
#         <Label value="115_gs" background="#B20D3C" />
#         <Label value="115_gs_w_motor" background="#f287f5" />
#         <Label value="115_la" background="#9936AC" />
#         <Label value="115_vt_1p" background="#F1463F" />
#         <Label value="115_vt_3p" background="#199588" />
#         <Label value="22_breaker" background="#3C0DB2" />
#         <Label value="22_cap_bank" background="#0DB24C" />
#         <Label value="22_ds" background="#7AB20D" />
#         <Label value="22_ds_out" background="#B2AA0A" />
#         <Label value="22_ds_la_out" background="#0DB20D" />
#         <Label value="22_gs" background="#0D0DB2" />
#         <Label value="22_ll" background="#16448c" />
#         <Label value="22_vt_1p" background="#3CB20D" />
#         <Label value="22_vt_3p" background="#1F9AEE" />
#         <Label value="BCU" background="#B88000" />
#         <Label value="DIM" background="#B20D0D" />
#         <Label value="DPM" background="#B24CCC" />
#         <Label value="LL" background="#0DB2B2" />
#         <Label value="MU" background="#0DACEF" />
#         <Label value="NGR" background="#B20DFF" />
#         <Label value="NGR_future" background="#B20DD0" />
#         <Label value="Q" background="#7A0DB2" />
#         <Label value="remote_io_module" background="#0DB2B2" />
#         <Label value="ss_man_mode" background="#F20D3C" />
#         <Label value="tele_protection" background="#0DACEF" />
#         <Label value="terminator_double" background="#0DB27A" />
#         <Label value="terminator_single" background="#0DB2B2" />
#         <Label value="terminator_splicing_kits" background="#FD7AB2" />
#         <Label value="terminator_w_future" background="#0D7AB2" />
#         <Label value="v_m" background="#0DB24C" />
#         <Label value="v_m_digital" background="#0DB27A" />
#     </RectangleLabels>
# </View>


async def main() -> None:
    prisma = Prisma()
    await prisma.connect()

    # write your queries here
    await prisma.component.create_many(
        data=[
            {
                "index": 1,
                "name": "11522_tx_dyn1",
                "color": "#6344B2",
                "part":"kv"
            },
            {
                "index": 2,
                "name": "11522_tx_ynyn0d1",
                "color": "#3C57B0",
            },
            {
                "index": 3,
                "name": "115_1way_ds_w_motor",
                "color": "#B2AA0A",
            },
            {
                "index": 4,
                "name": "115_3ways_ds",
                "color": "#B20D0D",
            },
            {
                "index": 5,
                "name": "115_3ways_ds_w_motor",
                "color": "#B20D0D",
            },
            {
                "index": 6,
                "name": "115_breaker",
                "color": "#F20D7A",
            },
            {
                "index": 7,
                "name": "115_buffer",
                "color": "#B20D0D",
            },
            {
                "index": 8,
                "name": "115_cvt_1p",
                "color": "#1CBCD2",
            },
            {
                "index": 9,
                "name": "115_cvt_3p",
                "color": "#E62A64",
            },
            {
                "index": 10,
                "name": "115_ds",
                "color": "#B20D0D",
            },
            {
                "index": 11,
                "name": "115_gs",
                "color": "#B20D3C",
            },
            {
                "index": 12,
                "name": "115_gs_w_motor",
                "color": "#f287f5",
            },
            {
                "index": 13,
                "name": "115_la",
                "color": "#9936AC",
            },
            {
                "index": 14,
                "name": "115_vt_1p",
                "color": "#F1463F",
            },
            {
                "index": 15,
                "name": "115_vt_3p",
                "color": "#199588",
            },
            {
                "index": 16,
                "name": "22_breaker",
                "color": "#3C0DB2",
            },
            {
                "index": 17,
                "name": "22_cap_bank",
                "color": "#0DB24C",
            },
            {
                "index": 18,
                "name": "22_ds",
                "color": "#7AB20D",
            },
            {
                "index": 19,
                "name": "22_ds_out",
                "color": "#B2AA0A",
            },
            {
                "index": 20,
                "name": "22_ds_la_out",
                "color": "#0DB20D",
            },
            {
                "index": 21,
                "name": "22_gs",
                "color": "#0D0DB2",
            },
            {
                "index": 22,
                "name": "22_ll",
                "color": "#16448c",
            },
            {
                "index": 23,
                "name": "22_vt_1p",
                "color": "#3CB20D",
            },
            {
                "index": 24,
                "name": "22_vt_3p",
                "color": "#1F9AEE",
            },
            {
                "index": 25,
                "name": "BCU",
                "color": "#B88000",
            },
            {
                "index": 26,
                "name": "DIM",
                "color": "#B20D0D",
            },
            {
                "index": 27,
                "name": "DPM",
                "color": "#B24CCC",
            },
            {
                "index": 28,
                "name": "LL",
                "color": "#0DB2B2",
            },
            {
                "index": 29,
                "name": "MU",
                "color": "#0DACEF",
            },
            {
                "index": 30,
                "name": "NGR",
                "color": "#B20DFF",
            },
            {
                "index": 31,
                "name": "NGR_future",
                "color": "#B20DD0",
            },
            {
                "index": 32,
                "name": "Q",
                "color": "#7A0DB2",
            },
            {
                "index": 33,
                "name": "remote_io_module",
                "color": "#0DB2B2",
            },
            {
                "index": 34,
                "name": "ss_man_mode",
                "color": "#F20D3C",
            },
            {
                "index": 35,
                "name": "tele_protection",
                "color": "#0DACEF",
            },
            {
                "index": 36,
                "name": "terminator_double",
                "color": "#0DB27A",
            },
            {
                "index": 37,
                "name": "terminator_single",
                "color": "#0DB2B2",
            },
            {
                "index": 38,
                "name": "terminator_splicing_kits",
                "color": "#FD7AB2",
            },
            {
                "index": 39,
                "name": "terminator_w_future",
                "color": "#0D7AB2",
            },
            {
                "index": 40,
                "name": "v_m",
                "color": "#0DB24C",
            },
            {
                "index": 41,
                "name": "v_m_digital",
                "color": "#0DB27A",
            },
        ]
    )

    await prisma.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
