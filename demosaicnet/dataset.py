"""Dataset loader for demosaicnet."""
import os
import subprocess
import shutil
import hashlib
import logging


import numpy as np
from imageio import imread
from torch.utils.data import Dataset as TorchDataset
import wget

from .mosaic import bayer, xtrans

__all__ = ["BAYER_MODE", "XTRANS_MODE", "Dataset",
           "TRAIN_SUBSET", "VAL_SUBSET", "TEST_SUBSET"]


log = logging.getLogger(__name__)

BAYER_MODE = "bayer"
"""Applies a Bayer mosaic pattern."""

XTRANS_MODE = "xtrans"
"""Applies an X-Trans mosaic pattern."""

TRAIN_SUBSET = "train"
"""Loads the 'train' subset of the data."""

VAL_SUBSET = "val"
"""Loads the 'val' subset of the data."""

TEST_SUBSET = "test"
"""Loads the 'test' subset of the data."""


class Dataset(TorchDataset):
    """Dataset of challenging image patches for demosaicking.

    Args:
        download(bool): if True, automatically download the dataset.
        mode(:class:`BAYER_MODE` or :class:`XTRANS_MODE`): mosaic pattern to apply to the data.
        subset(:class:`TRAIN_SUBET`, :class:`VAL_SUBSET` or :class:`TEST_SUBSET`): subset of the data to load.
    """

    def __init__(self, root, download=False,
                 mode=BAYER_MODE, subset="train"):

        super(Dataset, self).__init__()

        self.root = os.path.abspath(root)

        if subset not in [TRAIN_SUBSET, VAL_SUBSET, TEST_SUBSET]:
            raise ValueError("Dataset subet should be '%s', '%s' or '%s', got"
                             " %s" % (TRAIN_SUBSET, TEST_SUBSET, VAL_SUBSET,
                                      subset))

        if mode not in [BAYER_MODE, XTRANS_MODE]:
            raise ValueError("Dataset mode should be '%s' or '%s', got"
                             " %s" % (BAYER_MODE, XTRANS_MODE, mode))
        self.mode = mode

        listfile = os.path.join(self.root, subset, "filelist.txt")
        log.debug("Reading image list from %s", listfile)

        if not os.path.exists(listfile):
            if download:
                _download(self.root)
            else:
                log.error("Filelist %s not found", listfile)
                raise ValueError("Filelist %s not found" % listfile)
        else:
            log.debug("No need no download the data, filelist exists.")

        self.files = []
        with open(listfile, "r") as fid:
            for fname in fid.readlines():
                self.files.append(os.path.join(self.root, subset, fname.strip()))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Fetches a mosaic / demosaicked pair of images.

        Returns
            mosaic(np.array): with size [3, h, w] the mosaic data with separated color channels.
            img(np.array): with size [3, h, w] the groundtruth image.
        """
        fname = self.files[idx]
        img = np.array(imread(fname)).astype(np.float32) / (2**8-1)
        img = np.transpose(img, [2, 0, 1])

        if self.mode == BAYER_MODE:
            mosaic = bayer(img)
        else:
            mosaic = xtrans(img)

        return mosaic, img


CHECKSUMS = {
    'datasets.z01': 'da46277afe85d3a91c065e4751fb8175',
    'datasets.z02': 'e274a9646323d954b00094ea424e4e4c',
    'datasets.z03': 'e071cc595a99a5aa4545d06350e5165f',
    'datasets.z04': 'c3d2f229834569cd5ae6e2d1467c4a95',
    'datasets.z05': 'daf90136c7b1ee4bb4653e9b6bf4b67d',
    'datasets.z06': '87e85d2854d40116e066b28e5a8750cc',
    'datasets.z07': 'a0b2854bf025c87c0bfdf83ce9aa9055',
    'datasets.z08': '62125ccf29cd4b182dd81a4bb82f94c4',
    'datasets.z09': 'f990f8a5090d586f2f31e61b5e6434bd',
    'datasets.z10': '41ecf8d8b7d981604d661d258bf988db',
    'datasets.z100': '923a536ece64cd036eec4a13156531c8',
    'datasets.z101': '44a936558af2e830fdf65d9acb3960ab',
    'datasets.z102': 'b24870482b41200ab7b91f0bcd3ed718',
    'datasets.z103': 'a85521c1fe0b8d2d1a074b0b52bf9db1',
    'datasets.z104': 'aacc7a81ec9e9a7849e3a45b1cb12f7c',
    'datasets.z105': '19b62c0f0ae008b77df6465182f43dc4',
    'datasets.z106': '4b0c414ce5825a9e2249e5810f0e55f0',
    'datasets.z107': '7f6df7fea899a656fcde898225890daf',
    'datasets.z108': '16a877c357f112367200a2534b5e54f3',
    'datasets.z109': '9180129bf9c204184f729bdf1c284c9c',
    'datasets.z11': 'cff5b0e9950933fa9dd6ced8ffb9528f',
    'datasets.z110': 'a95a6fbfd32d90058b9e4b9f0645c646',
    'datasets.z111': 'f3c7894a7d04178ca417dd5ed3a9e649',
    'datasets.z112': 'd46a73703a72c07137424cad90c9c0bd',
    'datasets.z113': '04b3421e465c5ef8bd64fc23730b58f7',
    'datasets.z114': '9c31d2d94bd6f1ba321039b18c462175',
    'datasets.z115': '427a3a6f3f936b0ba435da35be3e4bc3',
    'datasets.z116': 'c633a66e7644d7d8e8148d651b76d93f',
    'datasets.z117': 'cf316d3acaf301fc7b2b7e250ef734dd',
    'datasets.z118': 'c3b53a604492499930dfd000d7d33fa4',
    'datasets.z119': 'e64ede2179c589abd9e587347d9ba3b0',
    'datasets.z12': '17b70298245ae7965f4e4b4fb01f19cc',
    'datasets.z120': 'ece09ee0bc30eab71f06716cca393029',
    'datasets.z121': '79db008803bf58593df6c32db8c0b3d0',
    'datasets.z122': '647b151eb30a44d123ce9ddfbb380094',
    'datasets.z123': '5065f755fe61c666d6ed28096e4047a3',
    'datasets.z124': '6306215855e30112c495291d5928e0b7',
    'datasets.z125': 'a55c6e31e7ad42170016a15791e25134',
    'datasets.z126': '582eb81f0251840507ca0b53e624b1e0',
    'datasets.z127': '3beefb769a01481a8bc7ae39bc2f539b',
    'datasets.z128': 'fac96b38f96ea364ea51020386597b5d',
    'datasets.z129': '3aaca8ce2b67d2c1fe764ae2b306d17f',
    'datasets.z13': '7ec2aa595441d9f698a46d707f299e8c',
    'datasets.z130': '9f94105dcc39cf5421b2df2532c06ec9',
    'datasets.z131': '5de2901388a2e531d6874cdaf23bf15e',
    'datasets.z132': '6851ed45004ae6864892532d1ad44b20',
    'datasets.z133': '0f6b417d57f9bdc9fa91d85ff5e3378d',
    'datasets.z134': '0602b14c8828f7a9fed92713047c695f',
    'datasets.z135': '3ab79fd4da5c4c5b5a7896a189ec43a7',
    'datasets.z136': 'dd05152db786d8189cdb419ac5d0018a',
    'datasets.z137': 'b4e97abef22ee8b81ac232760d9f539d',
    'datasets.z138': '4ce939f1fa4f3e110e989db07d53d33f',
    'datasets.z139': 'a096add471cc5e074852c063eec3863a',
    'datasets.z14': '08571b6629b8856813fb35b45bbc082c',
    'datasets.z140': '3d84e8cc84ab26969c5239be60222ad3',
    'datasets.z141': 'd77062f59ca9957d33c9a671657fd795',
    'datasets.z142': '82901d01917006348deb89aa37fe3629',
    'datasets.z143': '736f77856f0854b26fa951479691df8f',
    'datasets.z144': '55c44320975f4278a8837085c5e02eda',
    'datasets.z145': '087d3b7634bf4720a916767d5c6b7d70',
    'datasets.z146': '5659d6f0495dcdc5f5d98bf2efaaa09b',
    'datasets.z147': '66dd69b2f9348e3c0d0c93c3e61416dd',
    'datasets.z148': 'f3fc8f15aceb0f9bf04d786b894caa44',
    'datasets.z149': '3863be1d2b130f79399432cfc1281c2e',
    'datasets.z15': 'cc57e0c4466575436f670ac3e07ad2f3',
    'datasets.z150': '09750f2019da9ff7132b904b8bcbd895',
    'datasets.z151': 'b1573f086c0f7d1fdf249a8e3a9bb178',
    'datasets.z152': '1a2d4374aea1e22c0b676a6a7eac49ec',
    'datasets.z153': 'b24320708d2019ed71ab16055e971b1d',
    'datasets.z154': '7ba27e1946afa610e131f3afefe78326',
    'datasets.z155': '2f02c8b5470be4cb6b53e4c9e512394e',
    'datasets.z156': 'fa4f0977409f181820bf78174257d657',
    'datasets.z157': '6736b97a29d1393ec65ddc9376a06369',
    'datasets.z158': 'b4b72842b13ec3877bc530ca2470a0db',
    'datasets.z159': 'fa0dfa57c9d299175719bbbdf319c935',
    'datasets.z16': '02c213b708e2ee7ebd68464dfb2279fc',
    'datasets.z160': '6edbb9dc7fa6d12d2e21631ae14eaa8c',
    'datasets.z161': '4ab093ce5af2726e9ee71fdf1943e8e2',
    'datasets.z162': '4e21401db9f9884d953df20381c5fd97',
    'datasets.z163': '8d39f0ed1a1d9b5de22b583d00081522',
    'datasets.z164': '40fe425e2c5e89b87b44a6e9735590d5',
    'datasets.z165': '9552ff9b03e2dbb45befc9d1cc99ad81',
    'datasets.z166': '6c0098a36d6827aea846d8522c578751',
    'datasets.z167': 'ce6b8b981d92f5a61f2ec40089a400a2',
    'datasets.z168': '60f6e16a3e5e409a3fc89edd3e0034d5',
    'datasets.z169': '75eb975a10d5cbf136796651a1789b42',
    'datasets.z17': 'c92ae62205eaef02db27996f0dc6c282',
    'datasets.z170': '287966840fff015ed36da3a08a18ebfb',
    'datasets.z171': 'e08193b722af492a78ac36a3125ac8d9',
    'datasets.z172': '32c795461f194c38b25047faeb46fdb5',
    'datasets.z173': '58fbb396dbbb902ac2f2c43722573200',
    'datasets.z174': '506fee2b982ee81689f3fe4d89133cac',
    'datasets.z175': '9d553e31b07b23e30c427800168eec6b',
    'datasets.z176': '0f6e3048824ead093d3127434ac83a72',
    'datasets.z177': 'ece15c004fa708849295987b8b1aba9c',
    'datasets.z178': 'd9db14d92d56ae2970798417030b5bb4',
    'datasets.z179': '25ddbe866d0a6b9cebe8d90f7b801fa6',
    'datasets.z18': 'f55ddc31cf203f495e352182a5bbadc3',
    'datasets.z180': 'd2fc49c68c77d1da592aff4ab90c0915',
    'datasets.z181': '5a4a635b1f3535311c6caecf4ab3ba80',
    'datasets.z182': 'df51725daff3edfc377a5f6bc158ec3a',
    'datasets.z183': '626add199ec4f263ff278d5392f41c9c',
    'datasets.z184': '5069483fd064ee5e8c24a240e6ee7736',
    'datasets.z185': '589249e98db0a4ded1d3e4acefd07509',
    'datasets.z186': 'e4415c64463ef16bceb9d2e2fa934d71',
    'datasets.z187': 'e070cbaaf88a1085964244f6505c713c',
    'datasets.z188': '71b38eb51edff8b049a302bacbe344d6',
    'datasets.z189': '8fa7a8b58c9e7cb9e86bfd0ca5f6d2ea',
    'datasets.z19': '6c34cd0e39a33737983ebf89f6cabf5f',
    'datasets.z190': 'daad0ea7c87d0935e014a370c38cc926',
    'datasets.z191': 'c355e9ee9d0afa67faa34739b7f7cf79',
    'datasets.z192': 'c97d5a784625795cdf3c36c337986afe',
    'datasets.z193': '5f3b8425e215798c9e454cdbe586db90',
    'datasets.z194': '31b5d74c1cbbabbf58ee470467b40d12',
    'datasets.z195': '4c65958343bc2ea1a28e779ee7e5e498',
    'datasets.z196': '26ab3664e62c7fd5d0be673c45dd0d93',
    'datasets.z197': '32d690086b6e9f05e3ced3a126af870c',
    'datasets.z198': '323071827db89626c9f186455fbb38c9',
    'datasets.z199': '72475a8500be1ff21407a66f0e2e91a6',
    'datasets.z20': '4161d6eda0ca5ed9500f953f789a25b2',
    'datasets.z200': '44a863b9c9760cb87a23f1422f242c0d',
    'datasets.z201': 'dc38b455fa45e3ef0d5f06397507982e',
    'datasets.z202': 'b9ba231b317b008602f9472325b40e65',
    'datasets.z203': '36f4afc46258d80be626040956550028',
    'datasets.z204': '5c522fbdfe1f9d449c9189173f2ed2b2',
    'datasets.z205': '58d39995017eed2c4abcf9fcfd07e695',
    'datasets.z206': '2efdfc2abc834f0f0f1cabe10423f865',
    'datasets.z207': '04ead7536e5c13936c724f644ab1cb3a',
    'datasets.z208': '9e3e0a02a07bebcc7cdb62a0ad047946',
    'datasets.z209': 'c1fc44cd8b6f50955c8b3b317155ecb6',
    'datasets.z21': '8669b8bb9fa90628d4423c45648868b2',
    'datasets.z210': '210df79f8434bdb4e2a7d12c4078d972',
    'datasets.z211': 'f078d2d8a14b6c59f58c67865bbc3334',
    'datasets.z212': 'ef7d08a6cc39f6cb96b631ca61b440a0',
    'datasets.z213': '723057f7619d8820f142944f55f9542b',
    'datasets.z214': '2ba38ca8561b51710f660c03f84c0eb1',
    'datasets.z215': '92a6e97dfaff295110ddead242ebe932',
    'datasets.z216': '05f40901ae70f73b3c099fcdd4ca945e',
    'datasets.z217': 'b690ed3e8c6ba9f8bba9154d7e8f7ece',
    'datasets.z218': 'e290ca6f5573579df9f3aa7c5158891e',
    'datasets.z219': 'a8e4626968f089163179f30066ce732c',
    'datasets.z22': '0f90463abdc8f0f81d81249302cf2d09',
    'datasets.z220': '3ecd2c0c855505d2957046d784944fce',
    'datasets.z221': '6c9c28287fbadcab2ce777ef3134e5d6',
    'datasets.z222': '29361a77f05e5e68113fc23e11b54b4d',
    'datasets.z223': '9214257b9a87c0037e88709addba8948',
    'datasets.z224': '7596a516fb7e308f33a81c5b3c36810a',
    'datasets.z225': 'c1dc079f5261a976b1bd7f5c05cd4a02',
    'datasets.z226': '5b87815b0ccc5cacec83a399a52874aa',
    'datasets.z227': '6bed353dc50263b2c720af663c833bbc',
    'datasets.z228': '37ed3574bf978bccd6e2db9be00bae94',
    'datasets.z229': '8fd57367808fd77581f998850e5f935a',
    'datasets.z23': '90ae2cdcdc1663b80c20e080e5c0e038',
    'datasets.z230': '3f5ef3234da0236d2fbfaf7366407d70',
    'datasets.z231': 'f67d8320028620c8bdd9a800a78afa27',
    'datasets.z232': '5f831f25f8e8557168b38a7a28f8e7f9',
    'datasets.z233': '56ee8c4b01825ba7f12340ae8b990db3',
    'datasets.z234': '5428e98487b0e077cf9c24dc60599286',
    'datasets.z235': '883c0ea97facca4d57d5c9c54922e8be',
    'datasets.z236': 'e23fb6f610a3b528d5b310df4e452256',
    'datasets.z237': 'bd858e84a47668edc851dab131239ae4',
    'datasets.z238': 'db51bed3e3e5c6a40881f22532618533',
    'datasets.z239': 'c1be852117739fc63227a503b08a8436',
    'datasets.z24': '00dcb2e2a72b15a9aa9a646ecaea0019',
    'datasets.z240': 'f4e6da02349b03b4f433b3399dcf8b3c',
    'datasets.z241': '4f29898105aaf9f1a753a1c639947c2f',
    'datasets.z242': '168a15bd8367f5d5f3e5e8cf4d0da6af',
    'datasets.z243': '2930bced33bd1ecabce070fc831567e7',
    'datasets.z244': 'aed9c5ec05f57e3fa9e7b224d47fa7b0',
    'datasets.z245': '5aa83729fec805c166e48e5ec21530a5',
    'datasets.z246': '646ceabfae028d631568930b4056227a',
    'datasets.z247': '7381491175c1a63cc04ecac81148925b',
    'datasets.z248': '4064df81449c1980d0abaa8c7262b315',
    'datasets.z249': '7ae84d1dde2e935d86138d1e7b077df8',
    'datasets.z25': '5ae383bcd01d4ce22387680e28833f06',
    'datasets.z250': 'c018b41fbc4982a561b07cf0d52137f2',
    'datasets.z251': '9c9dc7a889d537fd1e02f4549529a5f1',
    'datasets.z252': 'b74df0680e7a62794902186fe1e3fec2',
    'datasets.z253': '894cbb618ddffe65ce2ada0f250ad79c',
    'datasets.z254': 'd5d8bc590d109c7d592d4df183c495a8',
    'datasets.z255': 'faf313a3edd70129c212d3dbd1de5042',
    'datasets.z256': '0fefcb84b66518df03a93ae53079409f',
    'datasets.z257': '887e8e8abaf09682903b9b1060fb8153',
    'datasets.z258': 'a700ae13abb7707032123468fab1bf55',
    'datasets.z259': 'cad7c974b832d27d1cfb4ae0e4dc6c3c',
    'datasets.z26': '5e0cdf281eeca969a4e0adfc44e11dcb',
    'datasets.z260': '5e1c11df40440e4e84354d40efe7940e',
    'datasets.z261': '2a75579c238569356855244d9fedf50b',
    'datasets.z262': '51d508271fef5558df387542b3561b67',
    'datasets.z263': 'e11b54dcd5069e838a34dcc2daebf4b5',
    'datasets.z264': '8a5a3b288d21a3c2ef641370d436703e',
    'datasets.z265': '56954047cba7c8732f0490323540af43',
    'datasets.z266': 'cad0325e494cc720c385ac7420acd2d7',
    'datasets.z267': '25de522b499ec7af12f583dd89a31769',
    'datasets.z268': '0e9882f392ca679e8c47276371384efd',
    'datasets.z269': '28487bc8eb731d4913254a9d63bb13ae',
    'datasets.z27': 'c78804dfac1e395156abd235ca416b33',
    'datasets.z270': '276fde25d412d8a1197e3dda307580d7',
    'datasets.z271': 'ba6f46558aa9ebc64efca2485b4f18ca',
    'datasets.z272': '5acceb08940d937d3023d98e745b8197',
    'datasets.z273': '57201a53390fe9c6a8c069397dcb81b8',
    'datasets.z274': '92491b6786ccea7b6ccccfb4e09c6d75',
    'datasets.z275': '349136c52b8554f03967d7083e5cd95c',
    'datasets.z276': '1525075dbf4d5d101d63cdade8bed9e7',
    'datasets.z277': 'eeb6365ef482cd2c6bac20aab8181081',
    'datasets.z278': 'ab1b04860b27fc11b7f57074a0815877',
    'datasets.z279': 'dc1cee0d4b69da9fd7aeb47f91768589',
    'datasets.z28': '0257b938256a2b7b55637970ebb3edcc',
    'datasets.z280': 'd168adf5e7c223a1d8dddfc663ebaeb0',
    'datasets.z281': '540fc1de91a90e9bd91b3f2b590ddbf6',
    'datasets.z282': 'f3fd6fbe05cfd53eb4a4c2e41bf75cc7',
    'datasets.z283': '20e89e60dd6a582bcb98b74df82699c3',
    'datasets.z284': '6e6ba6077437285881609999ead45463',
    'datasets.z285': 'fdc4bad8adb36b3d6653a438f1fc000f',
    'datasets.z286': '39c0f6fc7aace7e30a33e7e73afdb6ae',
    'datasets.z287': 'e60818eefd7426f3de0cc0746550be7f',
    'datasets.z288': 'd9db9107b8e92c0bf6a311a219363554',
    'datasets.z289': 'a250aa672d1f9165981cfaf1c6c8fff6',
    'datasets.z29': 'e1240710e1c4dd506aec03c02caf5606',
    'datasets.z290': '6dcb5a7674c927ec4e965a42d04a0ccd',
    'datasets.z291': 'c058fca515b7f714338816b72672ce20',
    'datasets.z292': '3fa254ed46ad6a6f7686836fe6fb7991',
    'datasets.z293': '782344c6620582d6b1681e142853a61e',
    'datasets.z294': '5263c5eb50cfec20adf82a89973a7547',
    'datasets.z295': '0c217819aa7308ce8f744511e572a632',
    'datasets.z296': 'cf8e7f1d3503ad6371ebcc9f827a29f8',
    'datasets.z297': 'f16861471be342b291e55991654c882b',
    'datasets.z298': 'ceb8b1170f1f2ec8113ef1c004df236c',
    'datasets.z299': '4f66a50c5dfb03143a6456c7fd925ec1',
    'datasets.z30': '75b0444187fc6c3df7bd3182108bc647',
    'datasets.z300': 'becbeea47cef192a1a13b35911c4795a',
    'datasets.z301': '92a67108c6bde11111f3b94f690f4b42',
    'datasets.z302': '06292210e05575cf1632a099648c16af',
    'datasets.z303': 'ecb7a48777719322c957ecc99340f04a',
    'datasets.z304': 'b318fc0b7d645d16467b78bbce95befa',
    'datasets.z305': '5f83e6a17e5977c2b99fc29893a1f479',
    'datasets.z306': 'dde2ec22363740081f54032a3add00e0',
    'datasets.z307': 'e0b7f9a7eddf2117a6127542883eb767',
    'datasets.z308': 'fff180911deaf4b476f42e6e47d78e6e',
    'datasets.z309': 'c55a8ac017f7ea69e77519fbcc617301',
    'datasets.z31': '04bee9374c7436d66f560bbbfc22299d',
    'datasets.z310': '3b4282fd1ab2b4f885df196a83726d34',
    'datasets.z311': '08a7c41c5d290750d9ebf6266a86bec8',
    'datasets.z312': 'b0f844ffd0ae6785e077ef8871dfc5da',
    'datasets.z313': 'bd03ba03a63877b17274e21ddb828218',
    'datasets.z314': '392f4528f9c345355434e7448a80b28b',
    'datasets.z315': 'e499cdf8acf1561720d4eb8f9ad9daad',
    'datasets.z316': '2575496c4d9c5082c6c4ef2f0cedeb69',
    'datasets.z317': '6f387284dbef478e02f7a5954da66015',
    'datasets.z318': '2ea47b6b7c9790bbfc2d3c238ffba391',
    'datasets.z319': '8e1e48b2aa1d6c7ae840a23360a3a8f8',
    'datasets.z32': 'ab930ee97741da82bfc778474f528a28',
    'datasets.z320': 'a785b37410ccf0366f97c0d221faa629',
    'datasets.z321': '16c73037fa0ca7704a3c4ceddbd7d599',
    'datasets.z322': 'af97027070cf5d057934dfd6bd819e61',
    'datasets.z323': '79078baf8d5fee935620f48aed2980a6',
    'datasets.z324': '987b626c8731b0092df593f0eaddb32a',
    'datasets.z325': 'fcc9a289015b40044c55ca96bc3dbe1e',
    'datasets.z326': 'f2921b782a23f2729e1a6641e8c99954',
    'datasets.z327': '6d1b488d88cf0fcaf5105b6544e5ea66',
    'datasets.z328': '268885eaeb6290be4ddfaefb34943985',
    'datasets.z329': '3951d3dd0527b54c5302720ea037cb12',
    'datasets.z33': 'fff4585bf34e5a7ed8f369b337aac901',
    'datasets.z330': '4b546d783366442d95eed64f882ae9f6',
    'datasets.z331': '4605615511ea901c18256306263b2226',
    'datasets.z332': '4127b2a8dd549b02ce3e465cfcfcc0a3',
    'datasets.z333': '711d282247b4fd56758c96c13bfe1b8e',
    'datasets.z34': '39b5d53f995fa8c223ed0d7a2de34652',
    'datasets.z35': 'ae351c1e2961f99a6d3ac37fbae27548',
    'datasets.z36': 'c2102c4a984d03c32c7c99378df953dc',
    'datasets.z37': '76ba463895984049a1814654b7290890',
    'datasets.z38': '5481901e75ee9d55066ba2731b2f36b7',
    'datasets.z39': '59815df91b2532d1e300bc71c976da12',
    'datasets.z40': '321ebe9b7812c14ee185fe2d6f16300c',
    'datasets.z41': '862bc3fcb9fc1df2bef61561bcca8090',
    'datasets.z42': '10303d540fea2150a7574cefcec92977',
    'datasets.z43': '2269db3212f2d2db86982408a2b24948',
    'datasets.z44': '80673bdbd722d02d97febeca00e57cf1',
    'datasets.z45': '42329dbb5c5165788902f33265db66a3',
    'datasets.z46': 'f81999194d418c515bb5df32c278d7f7',
    'datasets.z47': '2e3d3520636b5f3eb0cd2d649b6b4dab',
    'datasets.z48': 'e57e7aee104e80deaf068fbdd3292410',
    'datasets.z49': '36146291e4af0eff44e763e7e1facb4f',
    'datasets.z50': '60ef115c5c621c757b9b7075d5590c20',
    'datasets.z51': '02326e4392d176c2aa6d479cf43f29f8',
    'datasets.z52': 'b454f81ef7cda24cb13b8176c641d7ef',
    'datasets.z53': '95a4857fe7bc6230e6a3cc379085e989',
    'datasets.z54': '8ff35d1e9d738eb8b2afde04699ba73f',
    'datasets.z55': '75d19ea4a9a283b6d236e3353063d82d',
    'datasets.z56': '2ec9121fb364cc76eaf6fe7ff947dd04',
    'datasets.z57': 'f5d0b9b2b0a91f82dea784023f71cf3c',
    'datasets.z58': '79e318093d97bd573cc7aab4ed68dd6d',
    'datasets.z59': '68350ada72cfc22d2d6fc8ca636ffef2',
    'datasets.z60': '1ff9d36ec2b3723253503d524c90c9df',
    'datasets.z61': '9fb963355e8c895c1a0a02fa45eb6fb2',
    'datasets.z62': '5900cdc6c3d5f4b0c05825b47946096a',
    'datasets.z63': '9d039ddf86aef563bf7834faa2766e84',
    'datasets.z64': '15decde2fb2b4f869684f88f067378c1',
    'datasets.z65': '7cf283ebffd79c38bc09835caba483e1',
    'datasets.z66': 'e3b65f4facef36a355b444bd6b4ec73b',
    'datasets.z67': 'b007466dd2ac3388a5902fdf92979b2d',
    'datasets.z68': 'c07cd3fbc9a99e8f4d3cc9bbc55682e0',
    'datasets.z69': 'cff012a6a3af8a9f286cf68baf37ac73',
    'datasets.z70': '032ad7661e468723242b995357870a05',
    'datasets.z71': 'fd7aa78706b8ae7ff50eb2f312a14ea5',
    'datasets.z72': 'c2b90b8e8f75e36c927d523138d8ae93',
    'datasets.z73': '8c001e26b5f4ca0e061ff0b685d509c6',
    'datasets.z74': '484dcaf9311cc4315175e114acf01f32',
    'datasets.z75': '19f6896b426b99f34b99c88f3b68209f',
    'datasets.z76': '4a64c040ba9aa6e455fee774a08873e5',
    'datasets.z77': 'e96b1d94e51741ca916580c5f8287ef5',
    'datasets.z78': 'e0f5c9d93a3a5cfd95eeb61ee322650d',
    'datasets.z79': '75fbea4391cd38d85c583c0f504325f1',
    'datasets.z80': '7a99d15beff97f2dc620300f5ea82506',
    'datasets.z81': '1e303b839a0349a8497eb6c49e3ab4c3',
    'datasets.z82': '48c23365901b8d9cb51a20f7c71fa0a7',
    'datasets.z83': 'fed6445cb47f0b6cd6ed6f5482e38be7',
    'datasets.z84': '3d052276fc186ec151b5c237c5774e66',
    'datasets.z85': 'd84dccdf4fd4e6f6126b746d0519d19d',
    'datasets.z86': '772ab63f1ae266761db99124a152fc29',
    'datasets.z87': '44208b663adf97563018a416560cce6b',
    'datasets.z88': '6be7aaff3349239507b103c928afeaf5',
    'datasets.z89': '515daa16bcaf83b0be91dcc32e0e4985',
    'datasets.z90': 'f0fe0aa02bf0f19c0a7e301b532afa4c',
    'datasets.z91': '87fa807089240e01886389a6c4c77641',
    'datasets.z92': '1fd3154b3d8ec321f58f7685aedc5441',
    'datasets.z93': 'b24ddb4b3e2c0c1c96d7650a8320e446',
    'datasets.z94': 'b5e98cd90a629c4abb70d66a6976e0c7',
    'datasets.z95': '87c8d5d7f26e0fbf7422fda2194bef97',
    'datasets.z96': 'bb18c027a71f326a33b0a8fbe2d3a11f',
    'datasets.z97': '0489ed1c8b4fb7d5965d1565076d5c6a',
    'datasets.z98': 'a85303e08ce59fdf28f36ae9d0f20dcb',
    'datasets.z99': '992e63e126f05eca9fa9a84bbf66165c',
    'datasets.zip': '3434f60f5e9b263ef78e207b54e9debe',
}


def _download(dst):
    dst = os.path.abspath(dst)
    files = CHECKSUMS.keys()
    fullzip = os.path.join(dst, "datasets.zip")
    joinedzip = os.path.join(dst, "joined.zip")

    URL_ROOT = "https://data.csail.mit.edu/graphics/demosaicnet"

    if not os.path.exists(joinedzip):
        log.info("Dowloading %d files to %s (This will take a while, and ~80GB)", len(
            files), dst)

        os.makedirs(dst, exist_ok=True)
        for f in files:
            fname = os.path.join(dst, f)
            url = os.path.join(URL_ROOT, f)

            do_download = True
            if os.path.exists(fname):
                checksum = md5sum(fname)
                if checksum == CHECKSUMS[f]:  # File is is and correct
                    log.info('%s already downloaded, with correct checksum', f)
                    do_download = False
                else:
                    log.warning('%s checksums do not match, got %s, should be %s',
                                f, checksum, CHECKSUMS[f])
                    try:
                        os.remove(fname)
                    except OSError as e:
                        log.error("Could not delete broken part %s: %s", f, e)
                        raise ValueError

            if do_download:
                log.info('Downloading %s', f)
                wget.download(url, fname)

            checksum = md5sum(fname)

            if checksum == CHECKSUMS[f]:
                log.info("%s MD5 correct", f)
            else:
                log.error('%s checksums do not match, got %s, should be %s. Downloading failed',
                          f, checksum, CHECKSUMS[f])

        log.info("Joining zip files")
        cmd = " ".join(["zip", "-FF", fullzip, "--out", joinedzip])
        subprocess.check_call(cmd, shell=True)

        # Cleanup the parts
        for f in files:
            fname = os.path.join(dst, f)
            try:
                os.remove(fname)
            except OSError as e:
                log.warning("Could not delete file %s", f)

    # Extract
    wd = os.path.abspath(os.curdir)
    os.chdir(dst)
    log.info("Extracting files from %s", joinedzip)
    cmd = " ".join(["unzip", joinedzip])
    subprocess.check_call(cmd, shell=True)

    try:
        os.remove(joinedzip)
    except OSError as e:
        log.warning("Could not delete file %s", f)

    log.info("Moving subfolders")
    for k in ["train", "test", "val"]:
        shutil.move(os.path.join(dst, "images", k), os.path.join(dst, k))
    images = os.path.join(dst, "images")
    log.info("removing '%s' folder", images)
    shutil.rmtree(images)


def md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()
