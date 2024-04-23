# Pre-processing P.L.U.M.E. Van Data

This script was created to automatically pre-process P.L.U.M.E. Van data after multiple days of sampling

**Author:** Davi de Ferreyro Monticelli, iREACH group (University of British Columbia)

**Date:** 2023-06-17

**Version:** 1.0.0

P.L.U.M.E Dashboard currently (as of June 2023) records data from the:

- NOx monitor
- CO monitor
- O3 monitor
- 3D Sonic anemometer monitor (Wind direction, speed, vertical wind speed, and temperature)

In addition, the following monitors are installed in the mobile laboratory but are
not connected to the datalogger:

- UFP monitors (WCPC and FMPS)
- CO2 monitor
- BC monitor (microAethelometer)

However, some data require pre-processing, because:

- NOx: instrument is set to send a 1 Volt signal for every 10 ppb, however
       the maximum output is 2.5 V, meaning that if concentrations exceed
       250 ppb, a flat line will occur in the data.
       This setup (1V - 10ppb) can be changed in the instrument (check manual).

- CO:  due to limitations in the calibration process, the instrument records
       data with a zero = -0.494 ppm. This was verified with a collocation
       performed at Clark Drive monitoring station in 2022. Thus, all CO readings
       must be adjusted according to the results of this collocation.

- CO2: the voltage signal to the datalogger is not reliable. Attempts were made to
       properly convert the measured concentration to the voltage output, but unsuccessful.
       Thus, as an alternative, we sample using the LI-COR 850 software and incorporate
       the timeseries back in the P.L.U.M.E Dashboard sensor transcript.

- O3:  instrument is set to send a 1 Volt signal for every 10 ppb, however
       the maximum output is 2.5 V, meaning that if concentrations exceed
       250 ppb, a flat line will occur in the data. (But this never happens outdoors).
       This setup (1V - 10ppb) can be changed in the instrument (check manual).

- UFP: after several failed attempts to understand the pulse output of the instrument and
       why does it change for increasing steps in concentration, the WCPC was eventually
       disconnected from the datalogger. Thus, as an alternative, we sample using the TSI software
       and incorporate the timeseries back in the P.L.U.M.E Dashboard sensor transcript
       + laboratory tests indicate that inlet pressures below or above 0 alter concentration readings.
       this needs further attention before a code is implemented to fix it.

- BC:  we have not explored the connection of this instrument to the datalogger.
       Thus, as an alternative, we sample using the microAeth software and IN THE FUTURE will incorporate
       the timeseries back in the P.L.U.M.E Dashboard sensor transcript.

For such reasons, the functions below should save you time pre-processing all the P.L.U.M.E data
before you can run the post-processing scripts such as merge.py, baseline.py, peak.py etc.
also, any other script associated with P.L.U.M.E data but currently not integrated to P.L.U.M.E Dashboard
(e.g., AQ_and_EOI_Analysis.py script)
