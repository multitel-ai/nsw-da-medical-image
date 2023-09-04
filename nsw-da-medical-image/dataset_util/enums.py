"""enums: some classes to represent the dataset

All enums should be decorated with 'enum_by_index' so they have the methods

    def idx(self) -> int:
        ...
    def all_indices() -> list[int]:
        ...
    def from_idx(idx: int) -> T:
        ...

    def from_indices(indices: typing.Iterable[int]) -> list[T]:
        ...
"""

import random
import numpy as np

from sklearn.model_selection import KFold

from .util import EnumIdx


class FocalPlane(EnumIdx):
    F_n45 = "_F-45"
    F_n30 = "_F-30"
    F_n15 = "_F-15"
    F_0 = ""
    F_p15 = "_F15"
    F_p30 = "_F30"
    F_p45 = "_F45"

    @property
    def suffix(self) -> str:
        return self.value


class Phase(EnumIdx):
    tPB2 = "tPB2"
    tPNa = "tPNa"
    tPNf = "tPNf"
    t2 = "t2"
    t3 = "t3"
    t4 = "t4"
    t5 = "t5"
    t6 = "t6"
    t7 = "t7"
    t8 = "t8"
    t9p = "t9+"
    tM = "tM"
    tSB = "tSB"
    tB = "tB"
    tEB = "tEB"
    tHB = "tHB"
    beginning = "beginning"  # frames are before the first phase
    trailing = "trailing"  # some frames are past the last phase

    @property
    def label(self) -> str:
        return self.value


class Video(EnumIdx):
    "video: an identifier for each video of the dataset"

    @property
    def directory(self) -> str:
        return self.value[0]

    @property
    def n_frames(self) -> int:
        return self.value[1]

    @staticmethod
    def split(n_splits: int, seed: int = 42):
        """split the videos into multiple random set

        - n_splits: number of splits
        - seed: a pseudorandom seed for reproducibility
        """

        indices = Video.all_indices()
        random.Random(seed).shuffle(indices)

        count, truncated = divmod(len(Video), n_splits)

        if count == 0:
            raise ValueError(f"{n_splits=} is too much ({len(Video)=})")

        ranges = [
            (
                split_idx * count + min(split_idx, truncated),
                (split_idx + 1) * count + min(split_idx + 1, truncated),
            )
            for split_idx in range(n_splits)
        ]
        return [Video.from_indices(indices[lo:hi]) for lo, hi in ranges]

    AA83_7 = ("AA83-7", 285)
    AAL839_6 = ("AAL839-6", 557)
    AB028_6 = ("AB028-6", 291)
    AB91_1 = ("AB91-1", 558)
    AC264_1 = ("AC264-1", 342)
    ADM715_1_2 = ("ADM715-1-2", 446)
    AG274_2 = ("AG274-2", 561)
    AG782_6 = ("AG782-6", 565)
    AG782_8 = ("AG782-8", 565)
    AH988_4 = ("AH988-4", 551)
    AHS115_5 = ("AHS115-5", 557)
    AHS599_4 = ("AHS599-4", 464)
    AK383_1 = ("AK383-1", 458)
    AL702_9 = ("AL702-9", 549)
    AL884_2 = ("AL884-2", 560)
    ALR493_10 = ("ALR493-10", 566)
    ALR493_6 = ("ALR493-6", 566)
    AM33_2 = ("AM33-2", 562)
    AM365_7 = ("AM365-7", 459)
    AM685_3 = ("AM685-3", 362)
    AM716_1 = ("AM716-1", 572)
    AM716_7 = ("AM716-7", 572)
    AM918_2_5 = ("AM918-2-5", 555)
    AMT360_1_9 = ("AMT360-1-9", 394)
    AS1015_2 = ("AS1015-2", 370)
    AS556_3 = ("AS556-3", 559)
    AS563_1 = ("AS563-1", 202)
    AS662_2 = ("AS662-2", 466)
    AS71_3 = ("AS71-3", 204)
    BA1195_9 = ("BA1195-9", 662)
    BA25_7 = ("BA25-7", 462)
    BA258_6 = ("BA258-6", 436)
    BA518_7 = ("BA518-7", 552)
    BA560_1 = ("BA560-1", 574)
    BA560_2 = ("BA560-2", 574)
    BA782_2 = ("BA782-2", 376)
    BA958_2 = ("BA958-2", 360)
    BC167_4 = ("BC167-4", 287)
    BC254_1 = ("BC254-1", 561)
    BC254_5 = ("BC254-5", 561)
    BC277_10 = ("BC277-10", 457)
    BC277_9 = ("BC277-9", 457)
    BC396_1 = ("BC396-1", 471)
    BC396_2 = ("BC396-2", 471)
    BC518_6 = ("BC518-6", 551)
    BC750_7 = ("BC750-7", 554)
    BC782_2 = ("BC782-2", 461)
    BC973_4 = ("BC973-4", 555)
    BE327_2 = ("BE327-2", 572)
    BE340_9 = ("BE340-9", 469)
    BE413_3 = ("BE413-3", 417)
    BE645_10 = ("BE645-10", 557)
    BE645_3 = ("BE645-3", 557)
    BE645_7 = ("BE645-7", 557)
    BE735_2 = ("BE735-2", 215)
    BF924_3 = ("BF924-3", 557)
    BG201_1 = ("BG201-1", 558)
    BG723_5 = ("BG723-5", 555)
    BH034_4 = ("BH034-4", 564)
    BI104_3 = ("BI104-3", 271)
    BI224_2 = ("BI224-2", 422)
    BJ3371_10 = ("BJ3371-10", 413)
    BJ3371_9 = ("BJ3371-9", 413)
    BJ492_11 = ("BJ492-11", 561)
    BJ492_8 = ("BJ492-8", 560)
    BJ497_7 = ("BJ497-7", 460)
    BK428_2 = ("BK428-2", 359)
    BL042_8 = ("BL042-8", 564)
    BL285_1_3 = ("BL285-1-3", 567)
    BL366_6 = ("BL366-6", 557)
    BL418_7 = ("BL418-7", 567)
    BL526_1 = ("BL526-1", 656)
    BL526_4 = ("BL526-4", 656)
    BL526_5 = ("BL526-5", 656)
    BM007_9 = ("BM007-9", 559)
    BM016_2 = ("BM016-2", 557)
    BM016_5 = ("BM016-5", 557)
    BM076_4 = ("BM076-4", 554)
    BM201_1 = ("BM201-1", 569)
    BM209_8 = ("BM209-8", 277)
    BM256_1 = ("BM256-1", 555)
    BM256_4 = ("BM256-4", 555)
    BM352_6 = ("BM352-6", 567)
    BM414_3 = ("BM414-3", 413)
    BM655_10 = ("BM655-10", 576)
    BM968_3 = ("BM968-3", 362)
    BM984_2 = ("BM984-2", 218)
    BMH857_2 = ("BMH857-2", 456)
    BMS288_1 = ("BMS288-1", 284)
    BMS819_7 = ("BMS819-7", 568)
    BN1010_5 = ("BN1010-5", 552)
    BN356_3 = ("BN356-3", 423)
    BN356_6 = ("BN356-6", 423)
    BO613_7 = ("BO613-7", 564)
    BR953_7 = ("BR953-7", 563)
    BS1033_2 = ("BS1033-2", 557)
    BS1086_1 = ("BS1086-1", 569)
    BS294_7 = ("BS294-7", 201)
    BS495_9 = ("BS495-9", 560)
    BS544_1 = ("BS544-1", 281)
    BS596_5 = ("BS596-5", 208)
    BS648_2_4 = ("BS648-2-4", 516)
    BS648_7 = ("BS648-7", 516)
    BS777_8 = ("BS777-8", 754)
    BS836_11 = ("BS836-11", 545)
    BS918_6 = ("BS918-6", 562)
    BV646_6 = ("BV646-6", 154)
    BY829_1 = ("BY829-1", 453)
    BY829_3 = ("BY829-3", 453)
    CA063_10 = ("CA063-10", 552)
    CA063_6 = ("CA063-6", 552)
    CA364_7 = ("CA364-7", 552)
    CA390_2 = ("CA390-2", 555)
    CA390_6 = ("CA390-6", 555)
    CA658_12 = ("CA658-12", 568)
    CA658_6 = ("CA658-6", 568)
    CA704_2 = ("CA704-2", 559)
    CAV074_1 = ("CAV074-1", 555)
    CAV074_3 = ("CAV074-3", 555)
    CAV074_4 = ("CAV074-4", 555)
    CAV074_5 = ("CAV074-5", 555)
    CAV074_6 = ("CAV074-6", 555)
    CAV074_8 = ("CAV074-8", 555)
    CAV074_9 = ("CAV074-9", 555)
    CC007_2 = ("CC007-2", 561)
    CC336_9 = ("CC336-9", 347)
    CC455_3 = ("CC455-3", 557)
    CC563_5 = ("CC563-5", 457)
    CC751_4 = ("CC751-4", 352)
    CC938_4 = ("CC938-4", 421)
    CC966_1 = ("CC966-1", 563)
    CE417_1 = ("CE417-1", 468)
    CE417_3 = ("CE417-3", 468)
    CE417_6 = ("CE417-6", 468)
    CE525_2 = ("CE525-2", 572)
    CE604_1 = ("CE604-1", 204)
    CE712_2 = ("CE712-2", 571)
    CF946_6 = ("CF946-6", 353)
    CJ261_10 = ("CJ261-10", 542)
    CJ261_6 = ("CJ261-6", 542)
    CJ528_1 = ("CJ528-1", 457)
    CK601_2 = ("CK601-2", 544)
    CK601_4 = ("CK601-4", 544)
    CL783_2 = ("CL783-2", 555)
    CM010_10 = ("CM010-10", 550)
    CM010_7 = ("CM010-7", 550)
    CM1073_4 = ("CM1073-4", 207)
    CM146_1 = ("CM146-1", 403)
    CM627_1 = ("CM627-1", 551)
    CM627_3 = ("CM627-3", 551)
    CM627_8 = ("CM627-8", 551)
    CM641_1_8 = ("CM641-1-8", 562)
    CM782_8 = ("CM782-8", 425)
    CM892_5 = ("CM892-5", 567)
    CN473_1 = ("CN473-1", 558)
    CN606_5 = ("CN606-5", 478)
    CO119_8 = ("CO119-8", 547)
    CS552_2 = ("CS552-2", 552)
    CS552_4 = ("CS552-4", 552)
    CV306_2 = ("CV306-2", 550)
    CZ594_1 = ("CZ594-1", 553)
    CZ594_5 = ("CZ594-5", 553)
    DA1054_5 = ("DA1054-5", 549)
    DA309_5 = ("DA309-5", 565)
    DA684_4 = ("DA684-4", 651)
    DA769_1 = ("DA769-1", 198)
    DA781_4 = ("DA781-4", 584)
    DA925_6 = ("DA925-6", 553)
    DAS678_1 = ("DAS678-1", 420)
    DAS706_3 = ("DAS706-3", 548)
    DC307_1 = ("DC307-1", 647)
    DC307_2 = ("DC307-2", 647)
    DC307_7 = ("DC307-7", 647)
    DC932_2 = ("DC932-2", 203)
    DE069_10 = ("DE069-10", 565)
    DE069_2 = ("DE069-2", 565)
    DE069_3 = ("DE069-3", 565)
    DE069_7 = ("DE069-7", 565)
    DE604_3 = ("DE604-3", 557)
    DE842_3 = ("DE842-3", 473)
    DG487_4 = ("DG487-4", 279)
    DH1012_1 = ("DH1012-1", 546)
    DHDPI042_3 = ("DHDPI042-3", 556)
    DHDPI042_6 = ("DHDPI042-6", 556)
    DHDPI042_7 = ("DHDPI042-7", 556)
    DHDPI042_8 = ("DHDPI042-8", 556)
    DI358_3 = ("DI358-3", 214)
    DJC641_4 = ("DJC641-4", 651)
    DL020_3 = ("DL020-3", 549)
    DL266_4 = ("DL266-4", 204)
    DL61_2 = ("DL61-2", 279)
    DL617_6 = ("DL617-6", 658)
    DM1046_12 = ("DM1046-12", 537)
    DM1114_6 = ("DM1114-6", 285)
    DM235_3 = ("DM235-3", 421)
    DML271_2 = ("DML271-2", 268)
    DML373_2 = ("DML373-2", 471)
    DN340_2 = ("DN340-2", 544)
    DN376_8 = ("DN376-8", 559)
    DN881_6 = ("DN881-6", 363)
    DRL1048_1 = ("DRL1048-1", 651)
    DS17_2 = ("DS17-2", 429)
    DS61_1 = ("DS61-1", 367)
    DS666_9 = ("DS666-9", 562)
    DS947_2 = ("DS947-2", 280)
    DSE41_2 = ("DSE41-2", 494)
    DSM138_5 = ("DSM138-5", 572)
    DT336_1 = ("DT336-1", 479)
    DT336_2 = ("DT336-2", 479)
    DV116_3 = ("DV116-3", 352)
    DV210_4 = ("DV210-4", 546)
    DV210_8 = ("DV210-8", 546)
    DV305_3 = ("DV305-3", 346)
    DV728_6 = ("DV728-6", 548)
    DV728_7 = ("DV728-7", 548)
    DY236_4 = ("DY236-4", 411)
    EA32_6 = ("EA32-6", 413)
    EC234_5 = ("EC234-5", 374)
    EE656_3 = ("EE656-3", 469)
    EH309_8 = ("EH309-8", 549)
    EH315_3 = ("EH315-3", 541)
    EH315_8 = ("EH315-8", 541)
    EH512_10 = ("EH512-10", 144)
    EJ393_3 = ("EJ393-3", 208)
    FA344_5 = ("FA344-5", 492)
    FA662_6 = ("FA662-6", 541)
    FC048_6 = ("FC048-6", 364)
    FC1164_11 = ("FC1164-11", 486)
    FD156_4 = ("FD156-4", 555)
    FE14_020 = ("FE14-020", 357)
    FF717_4 = ("FF717-4", 549)
    FH658_4 = ("FH658-4", 567)
    FM1017_5 = ("FM1017-5", 454)
    FM162_6 = ("FM162-6", 547)
    FM864_7 = ("FM864-7", 273)
    FN852_1 = ("FN852-1", 646)
    FS987_5 = ("FS987-5", 154)
    FV709_11 = ("FV709-11", 367)
    GA1087_6 = ("GA1087-6", 556)
    GA122_8 = ("GA122-8", 570)
    GA365_1 = ("GA365-1", 145)
    GA425_1 = ("GA425-1", 559)
    GA664_1 = ("GA664-1", 576)
    GA664_3 = ("GA664-3", 576)
    GA664_4 = ("GA664-4", 576)
    GA664_8 = ("GA664-8", 576)
    GA703_2_7 = ("GA703-2-7", 458)
    GA800_4 = ("GA800-4", 471)
    GA817_1_8 = ("GA817-1-8", 478)
    GA911_3 = ("GA911-3", 211)
    GA982_7 = ("GA982-7", 467)
    GC340_1 = ("GC340-1", 554)
    GC340_10 = ("GC340-10", 554)
    GC340_3 = ("GC340-3", 554)
    GC381_10 = ("GC381-10", 563)
    GC658_3 = ("GC658-3", 545)
    GC658_9 = ("GC658-9", 545)
    GC702_6 = ("GC702-6", 458)
    GC836_1 = ("GC836-1", 471)
    GC836_4 = ("GC836-4", 471)
    GC851_5 = ("GC851-5", 355)
    GD391_8 = ("GD391-8", 559)
    GD391_9 = ("GD391-9", 559)
    GE1055_6 = ("GE1055-6", 448)
    GE218_3 = ("GE218-3", 1140)
    GE294_4 = ("GE294-4", 407)
    GE663_5 = ("GE663-5", 473)
    GE843_2 = ("GE843-2", 212)
    GF083_5 = ("GF083-5", 263)
    GF1042_1_3 = ("GF1042-1-3", 552)
    GF1042_2_6 = ("GF1042-2-6", 472)
    GF667_1_1 = ("GF667-1-1", 559)
    GF667_2_6 = ("GF667-2-6", 549)
    GF976_4 = ("GF976-4", 420)
    GG677_5 = ("GG677-5", 659)
    GJ165_5 = ("GJ165-5", 542)
    GJ191_1 = ("GJ191-1", 556)
    GJ285_1 = ("GJ285-1", 367)
    GJ316_1 = ("GJ316-1", 566)
    GM213_2 = ("GM213-2", 192)
    GM293_2 = ("GM293-2", 570)
    GM293_3 = ("GM293-3", 570)
    GM456_3 = ("GM456-3", 363)
    GM537_7 = ("GM537-7", 554)
    GM858_1_3 = ("GM858-1-3", 454)
    GM858_2_6 = ("GM858-2-6", 454)
    GML002_7 = ("GML002-7", 478)
    GNB477_5 = ("GNB477-5", 422)
    GRSO424_8 = ("GRSO424-8", 477)
    GS205_2 = ("GS205-2", 546)
    GS205_6 = ("GS205-6", 546)
    GS220_2 = ("GS220-2", 357)
    GS334_6 = ("GS334-6", 547)
    GS349_4 = ("GS349-4", 426)
    GS400_7 = ("GS400-7", 558)
    GS415_5 = ("GS415-5", 545)
    GS430_1 = ("GS430-1", 547)
    GS430_2 = ("GS430-2", 547)
    GS430_9 = ("GS430-9", 547)
    GS490_2 = ("GS490-2", 557)
    GS490_4 = ("GS490-4", 557)
    GS490_7 = ("GS490-7", 557)
    GS490__6 = ("GS490-_6", 557)
    GS611_3 = ("GS611-3", 566)
    GS611_4 = ("GS611-4", 566)
    GS753_8 = ("GS753-8", 553)
    GS811_3 = ("GS811-3", 563)
    GS826_2 = ("GS826-2", 546)
    GS955_7 = ("GS955-7", 565)
    GS980_2 = ("GS980-2", 563)
    GSS052_2 = ("GSS052-2", 450)
    GSS052_6 = ("GSS052-6", 450)
    GT353_3 = ("GT353-3", 412)
    GUDPI077_8 = ("GUDPI077-8", 342)
    HA1040_4 = ("HA1040-4", 573)
    HC459_6 = ("HC459-6", 545)
    HC724_5 = ("HC724-5", 479)
    HE377_4 = ("HE377-4", 562)
    HE444_3 = ("HE444-3", 554)
    HE444_4 = ("HE444-4", 554)
    HE469_5 = ("HE469-5", 560)
    HF71_3 = ("HF71-3", 562)
    HH569_2 = ("HH569-2", 568)
    HH569_4 = ("HH569-4", 568)
    HL369_6 = ("HL369-6", 361)
    HM214_2_9 = ("HM214-2-9", 560)
    HM69_4 = ("HM69-4", 553)
    HS15_11 = ("HS15-11", 363)
    HS160_4 = ("HS160-4", 458)
    HS893_8 = ("HS893-8", 560)
    HV298_8 = ("HV298-8", 417)
    IZ1001_1 = ("IZ1001-1", 544)
    JC858_4 = ("JC858-4", 548)
    JE021_4 = ("JE021-4", 555)
    JL073_6 = ("JL073-6", 548)
    JM1088_1 = ("JM1088-1", 548)
    JS292_7 = ("JS292-7", 572)
    JS292_8 = ("JS292-8", 572)
    JV227_2 = ("JV227-2", 562)
    JV227_5 = ("JV227-5", 562)
    KA474_5 = ("KA474-5", 478)
    KD774_4 = ("KD774-4", 492)
    KF460_11 = ("KF460-11", 480)
    KF460_3 = ("KF460-3", 480)
    KF460_4 = ("KF460-4", 480)
    KF460_5 = ("KF460-5", 480)
    KF460_7 = ("KF460-7", 480)
    KJ1077_3 = ("KJ1077-3", 570)
    KJ426_9 = ("KJ426-9", 215)
    KT573_4 = ("KT573-4", 413)
    LA1012_5 = ("LA1012-5", 459)
    LA1071_6 = ("LA1071-6", 564)
    LA367_4 = ("LA367-4", 565)
    LA386_4 = ("LA386-4", 562)
    LA467_2 = ("LA467-2", 541)
    LA733_3 = ("LA733-3", 216)
    LA825_5 = ("LA825-5", 553)
    LAC959_6 = ("LAC959-6", 559)
    LBE649_3 = ("LBE649-3", 449)
    LBE857_1 = ("LBE857-1", 155)
    LBM519_1 = ("LBM519-1", 545)
    LBM519_10 = ("LBM519-10", 545)
    LBM659_6 = ("LBM659-6", 405)
    LBR602_1 = ("LBR602-1", 212)
    LBS371_1_8 = ("LBS371-1-8", 647)
    LC161_1_4 = ("LC161-1-4", 457)
    LC161_2_5 = ("LC161-2-5", 457)
    LC47_8 = ("LC47-8", 542)
    LC498_2 = ("LC498-2", 543)
    LC648_1 = ("LC648-1", 477)
    LC648_2 = ("LC648-2", 477)
    LC765_2 = ("LC765-2", 209)
    LCF544_2 = ("LCF544-2", 415)
    LCF979_1 = ("LCF979-1", 560)
    LD400_1 = ("LD400-1", 473)
    LD400_6 = ("LD400-6", 473)
    LE679_8 = ("LE679-8", 415)
    LEG557_3 = ("LEG557-3", 412)
    LFA766_1 = ("LFA766-1", 204)
    LG168_3 = ("LG168-3", 559)
    LGA21_6 = ("LGA21-6", 410)
    LGA881_1_2 = ("LGA881-1-2", 210)
    LGA881_2_5 = ("LGA881-2-5", 369)
    LH1169_8 = ("LH1169-8", 550)
    LHV745_7 = ("LHV745-7", 561)
    LK523_2 = ("LK523-2", 206)
    LK584_2 = ("LK584-2", 556)
    LK584_3 = ("LK584-3", 556)
    LL1196_5 = ("LL1196-5", 568)
    LL854_1 = ("LL854-1", 451)
    LLA399_2 = ("LLA399-2", 419)
    LLN757_6 = ("LLN757-6", 547)
    LLN873_1 = ("LLN873-1", 550)
    LM184_3 = ("LM184-3", 416)
    LM184_4 = ("LM184-4", 416)
    LM335_7 = ("LM335-7", 555)
    LM844_1 = ("LM844-1", 217)
    LM985_4 = ("LM985-4", 211)
    LMMG218_1_10 = ("LMMG218-1-10", 558)
    LN233_3 = ("LN233-3", 421)
    LNA592_8 = ("LNA592-8", 547)
    LNA592_9 = ("LNA592-9", 547)
    LP181_1 = ("LP181-1", 363)
    LP284_3 = ("LP284-3", 460)
    LS058_7 = ("LS058-7", 565)
    LS058_8 = ("LS058-8", 565)
    LS1035_1 = ("LS1035-1", 556)
    LS1045_4 = ("LS1045-4", 549)
    LS123_3 = ("LS123-3", 548)
    LS359_1_7 = ("LS359-1-7", 393)
    LS366_1 = ("LS366-1", 209)
    LS93_8 = ("LS93-8", 584)
    LSD500_3 = ("LSD500-3", 568)
    LSD500_6 = ("LSD500-6", 568)
    LT1112_5 = ("LT1112-5", 581)
    LT634_4 = ("LT634-4", 559)
    LTA908_2 = ("LTA908-2", 429)
    LTE064_1 = ("LTE064-1", 561)
    LTE064_5 = ("LTE064-5", 561)
    LTE064_8 = ("LTE064-8", 561)
    LV366_6 = ("LV366-6", 460)
    LV488_7 = ("LV488-7", 564)
    LV613_2 = ("LV613-2", 467)
    LV683_2_3 = ("LV683-2-3", 559)
    LV683_2_8 = ("LV683-2-8", 559)
    LV723_1 = ("LV723-1", 563)
    LV723_9 = ("LV723-9", 563)
    LYI1079_2 = ("LYI1079-2", 208)
    LZ865_2 = ("LZ865-2", 206)
    MA1007_3 = ("MA1007-3", 204)
    MA1059_3 = ("MA1059-3", 461)
    MA470_5 = ("MA470-5", 275)
    MA488_3 = ("MA488-3", 574)
    MA505_1 = ("MA505-1", 554)
    MA505_2 = ("MA505-2", 554)
    MA595_5 = ("MA595-5", 474)
    MA752_1 = ("MA752-1", 539)
    MA797_4 = ("MA797-4", 563)
    MA797_7 = ("MA797-7", 563)
    MA8_2 = ("MA8-2", 214)
    MA885_10 = ("MA885-10", 561)
    MAS094_2 = ("MAS094-2", 560)
    MAS094_5 = ("MAS094-5", 560)
    MAS203_4 = ("MAS203-4", 556)
    MAS203_6 = ("MAS203-6", 556)
    MC373_5 = ("MC373-5", 489)
    MC427_1 = ("MC427-1", 210)
    MC663_7 = ("MC663-7", 567)
    MC710_3 = ("MC710-3", 288)
    MC833_6 = ("MC833-6", 143)
    MC933_2 = ("MC933-2", 559)
    MDCH869_4 = ("MDCH869-4", 541)
    ME378_4 = ("ME378-4", 418)
    ME540_11 = ("ME540-11", 545)
    ME540_7 = ("ME540-7", 545)
    ME577_7 = ("ME577-7", 505)
    ME799_5 = ("ME799-5", 575)
    MF532_3 = ("MF532-3", 371)
    MG1147_6 = ("MG1147-6", 563)
    MI820_4 = ("MI820-4", 552)
    MJ402_10 = ("MJ402-10", 559)
    ML585_2 = ("ML585-2", 566)
    ML954_3 = ("ML954-3", 554)
    MM1134_3 = ("MM1134-3", 484)
    MM334_5 = ("MM334-5", 669)
    MM41_7 = ("MM41-7", 422)
    MM445_2_1 = ("MM445-2-1", 565)
    MM445_2_2 = ("MM445-2-2", 565)
    MM445_2_9 = ("MM445-2-9", 565)
    MM834_5 = ("MM834-5", 539)
    MM84_8 = ("MM84-8", 283)
    MM897_7 = ("MM897-7", 477)
    MM912_4 = ("MM912-4", 212)
    MN011_3 = ("MN011-3", 464)
    MN32_6 = ("MN32-6", 561)
    MP228_5 = ("MP228-5", 479)
    MRA165_6 = ("MRA165-6", 544)
    MRA165_7T = ("MRA165-7T", 544)
    MS1034_1 = ("MS1034-1", 421)
    MS511_2_3 = ("MS511-2-3", 550)
    MS565_10 = ("MS565-10", 459)
    MS624_4 = ("MS624-4", 565)
    MS624_6 = ("MS624-6", 565)
    MS624_7 = ("MS624-7", 565)
    MS624_8 = ("MS624-8", 565)
    MT351_4 = ("MT351-4", 281)
    MT520_4 = ("MT520-4", 542)
    MV750_5 = ("MV750-5", 215)
    MV930_2 = ("MV930-2", 211)
    NA834_7 = ("NA834-7", 543)
    NC636_4 = ("NC636-4", 467)
    ND1068_3 = ("ND1068-3", 206)
    NE429_4 = ("NE429-4", 543)
    NK206_3 = ("NK206-3", 196)
    OA170_11 = ("OA170-11", 557)
    OA333_6 = ("OA333-6", 567)
    OC110_5 = ("OC110-5", 351)
    OF387_2 = ("OF387-2", 560)
    OF960_2 = ("OF960-2", 556)
    OJ319_10 = ("OJ319-10", 465)
    OJ319_2 = ("OJ319-2", 465)
    OJ319_3 = ("OJ319-3", 465)
    OJ319_5 = ("OJ319-5", 465)
    OJ319_6 = ("OJ319-6", 465)
    OJ319_7 = ("OJ319-7", 465)
    OJ319_8 = ("OJ319-8", 465)
    OJ319_9 = ("OJ319-9", 465)
    OP517_1 = ("OP517-1", 563)
    PA1217_8 = ("PA1217-8", 560)
    PA145_1 = ("PA145-1", 472)
    PA145_2 = ("PA145-2", 472)
    PA214_5 = ("PA214-5", 562)
    PA276_3 = ("PA276-3", 416)
    PA289_8 = ("PA289-8", 564)
    PA337_4 = ("PA337-4", 279)
    PA731_2_3 = ("PA731-2-3", 553)
    PA745_3 = ("PA745-3", 553)
    PA916_1_10 = ("PA916-1-10", 297)
    PAS742_3 = ("PAS742-3", 415)
    PC55_2 = ("PC55-2", 559)
    PC758_2 = ("PC758-2", 208)
    PC809_7 = ("PC809-7", 561)
    PD496_6 = ("PD496-6", 556)
    PE081_5 = ("PE081-5", 567)
    PE256_2 = ("PE256-2", 535)
    PE256_5 = ("PE256-5", 535)
    PE724_1 = ("PE724-1", 213)
    PE83_6 = ("PE83-6", 569)
    PE863_4 = ("PE863-4", 558)
    PE863_9 = ("PE863-9", 558)
    PG209_3 = ("PG209-3", 413)
    PH394_2 = ("PH394-2", 567)
    PH664_7 = ("PH664-7", 352)
    PH783_3 = ("PH783-3", 563)
    PI1004_3 = ("PI1004-3", 553)
    PI1027_1 = ("PI1027-1", 553)
    PJ533_8 = ("PJ533-8", 145)
    PL974_9 = ("PL974-9", 550)
    PM273_6 = ("PM273-6", 554)
    PMDPI029_1_1 = ("PMDPI029-1-1", 559)
    PMDPI029_1_10 = ("PMDPI029-1-10", 559)
    PMDPI029_1_11 = ("PMDPI029-1-11", 559)
    PMDPI029_1_2 = ("PMDPI029-1-2", 559)
    PMDPI029_1_3 = ("PMDPI029-1-3", 559)
    PMDPI029_1_5 = ("PMDPI029-1-5", 559)
    PMDPI029_1_6 = ("PMDPI029-1-6", 559)
    PMDPI029_1_8 = ("PMDPI029-1-8", 559)
    PMDPI029_1_9 = ("PMDPI029-1-9", 559)
    PN110_11 = ("PN110-11", 484)
    PN110_9 = ("PN110-9", 484)
    PN636_1_6 = ("PN636-1-6", 476)
    PO13_3 = ("PO13-3", 423)
    PS148_9 = ("PS148-9", 570)
    PS292_4 = ("PS292-4", 463)
    PV361_2 = ("PV361-2", 565)
    QA374_7 = ("QA374-7", 137)
    QC211_2 = ("QC211-2", 549)
    QC211_6 = ("QC211-6", 549)
    QC267_7 = ("QC267-7", 560)
    QC267_8 = ("QC267-8", 560)
    QC697_4 = ("QC697-4", 561)
    QD472_3 = ("QD472-3", 347)
    RA361_4 = ("RA361-4", 485)
    RA467_2 = ("RA467-2", 275)
    RA580_2 = ("RA580-2", 548)
    RA803_2 = ("RA803-2", 560)
    RA803_4 = ("RA803-4", 560)
    RBC697_1 = ("RBC697-1", 552)
    RC1103_1 = ("RC1103-1", 556)
    RC54_4 = ("RC54-4", 562)
    RC545_2_5 = ("RC545-2-5", 558)
    RC545_2_8 = ("RC545-2-8", 558)
    RC545_2_9 = ("RC545-2-9", 558)
    RC755_1 = ("RC755-1", 540)
    RC755_3 = ("RC755-3", 540)
    RC755_4 = ("RC755-4", 540)
    RC755_6 = ("RC755-6", 540)
    RC755_7 = ("RC755-7", 540)
    RC755_9 = ("RC755-9", 540)
    RC812_3 = ("RC812-3", 552)
    RD1142_2 = ("RD1142-2", 283)
    RD167_7 = ("RD167-7", 279)
    RE260_6 = ("RE260-6", 556)
    RE828_1 = ("RE828-1", 460)
    RG434_11 = ("RG434-11", 557)
    RG434__10 = ("RG434-_10", 557)
    RG864_1 = ("RG864-1", 355)
    RG944_2 = ("RG944-2", 568)
    RI273_6 = ("RI273-6", 548)
    RI382_2 = ("RI382-2", 561)
    RK787_3 = ("RK787-3", 210)
    RL461_4 = ("RL461-4", 425)
    RL747_8 = ("RL747-8", 558)
    RL948_2 = ("RL948-2", 373)
    RLFS800_2 = ("RLFS800-2", 558)
    RM126_1 = ("RM126-1", 554)
    RM126_10 = ("RM126-10", 554)
    RM126_11 = ("RM126-11", 554)
    RM126_2 = ("RM126-2", 554)
    RM126_4 = ("RM126-4", 554)
    RM126_5 = ("RM126-5", 554)
    RM126_6 = ("RM126-6", 554)
    RM126_7 = ("RM126-7", 554)
    RM126_8 = ("RM126-8", 554)
    RM126_9 = ("RM126-9", 554)
    RM184_1 = ("RM184-1", 561)
    RM29_5 = ("RM29-5", 289)
    RM549_1 = ("RM549-1", 377)
    RM855_3 = ("RM855-3", 553)
    RMN410_3 = ("RMN410-3", 550)
    RO793_2 = ("RO793-2", 483)
    RS362_4 = ("RS362-4", 421)
    RS363_7 = ("RS363-7", 557)
    RS781_7 = ("RS781-7", 660)
    RV146N2_6 = ("RV146N2-6", 264)
    RV454_6 = ("RV454-6", 556)
    RV754_4 = ("RV754-4", 466)
    SA288_6 = ("SA288-6", 555)
    SC385_11 = ("SC385-11", 552)
    SC385_9 = ("SC385-9", 552)
    SC700_1 = ("SC700-1", 365)
    SC818_2 = ("SC818-2", 208)
    SDSA208_4 = ("SDSA208-4", 540)
    SE040_4 = ("SE040-4", 277)
    SHE580_7 = ("SHE580-7", 416)
    SK308_10 = ("SK308-10", 558)
    SK308_7 = ("SK308-7", 558)
    SK902_1_8 = ("SK902-1-8", 523)
    SK902_6 = ("SK902-6", 559)
    SL313_11 = ("SL313-11", 642)
    SLM044_1_1 = ("SLM044-1-1", 560)
    SM307_1_12 = ("SM307-1-12", 566)
    SM307_1_9 = ("SM307-1-9", 566)
    SM686_7 = ("SM686-7", 361)
    SN586_8 = ("SN586-8", 398)
    SS527_8 = ("SS527-8", 556)
    SS684_8 = ("SS684-8", 551)
    SS722_8 = ("SS722-8", 555)
    ST586_7 = ("ST586-7", 554)
    TA12_1 = ("TA12-1", 207)
    TA239_2 = ("TA239-2", 557)
    TA656_2 = ("TA656-2", 469)
    TA757_4 = ("TA757-4", 476)
    TA757_9 = ("TA757-9", 476)
    TAS1069_2 = ("TAS1069-2", 205)
    TC1047_2 = ("TC1047-2", 486)
    TD958_2_1 = ("TD958-2-1", 369)
    TH481_5 = ("TH481-5", 552)
    TJ297_4 = ("TJ297-4", 420)
    TJ899_2 = ("TJ899-2", 547)
    TK319_10 = ("TK319-10", 566)
    TL179_5 = ("TL179-5", 421)
    TL475_7 = ("TL475-7", 418)
    TM272_9 = ("TM272-9", 248)
    TM294_2 = ("TM294-2", 552)
    TM312_2 = ("TM312-2", 559)
    TM312_6 = ("TM312-6", 559)
    TM428_3 = ("TM428-3", 214)
    TM981_1 = ("TM981-1", 553)
    TN359_10 = ("TN359-10", 545)
    TN359_9 = ("TN359-9", 545)
    TN611_7 = ("TN611-7", 414)
    TN807_3 = ("TN807-3", 287)
    TN888_3 = ("TN888-3", 145)
    TT615_1_8 = ("TT615-1-8", 561)
    TV1166_4 = ("TV1166-4", 473)
    TV654_4 = ("TV654-4", 574)
    UL050__10 = ("UL050-_10", 568)
    UL050__9 = ("UL050-_9", 568)
    VA197_2 = ("VA197-2", 411)
    VA197_5 = ("VA197-5", 411)
    VA225_6 = ("VA225-6", 410)
    VA95_1 = ("VA95-1", 572)
    VC104_2 = ("VC104-2", 559)
    VC581_1 = ("VC581-1", 456)
    VC581_10 = ("VC581-10", 456)
    VC581_11 = ("VC581-11", 456)
    VC581_12 = ("VC581-12", 456)
    VC581_2 = ("VC581-2", 456)
    VC581_3 = ("VC581-3", 456)
    VC581_5 = ("VC581-5", 456)
    VC581_6 = ("VC581-6", 456)
    VC581_7 = ("VC581-7", 456)
    VC789_3 = ("VC789-3", 359)
    VF269_7 = ("VF269-7", 553)
    VH99_3 = ("VH99-3", 472)
    VM195_5 = ("VM195-5", 416)
    VM195_6 = ("VM195-6", 416)
    VM569_7 = ("VM569-7", 570)
    VM570_4 = ("VM570-4", 552)
    VM570_8 = ("VM570-8", 552)
    VN484_1 = ("VN484-1", 654)
    VS321_6 = ("VS321-6", 463)
    VS321_7 = ("VS321-7", 463)
    VS510_2 = ("VS510-2", 566)
    VS510_7 = ("VS510-7", 566)
    WA1014_3 = ("WA1014-3", 554)
    WA402_7 = ("WA402-7", 558)
    WM472_8 = ("WM472-8", 555)
    WS1048_4 = ("WS1048-4", 430)
    WS531_4 = ("WS531-4", 432)
    ZL1077_1 = ("ZL1077-1", 384)
    ZS435_5 = ("ZS435-5", 558)
    ZS435_6 = ("ZS435-6", 558)


"""to generate the content, run this code
import pathlib
f15_dir = pathlib.Path("nsw-dataset-processed-all/embryo_dataset_F15")
videos = sorted(list(f15_dir.iterdir()))
for vid in videos:
    count = len(list(vid.iterdir()))
    val = vid.name
    key = val.replace("-", "_")
    print(f"    {key} = (\"{val}\", {count})")
"""
