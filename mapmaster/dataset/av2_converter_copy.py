from functools import partial
from multiprocessing import Pool
import multiprocessing
from random import sample
import time
import mmcv
import logging
from pathlib import Path
from os import path as osp
import os
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.lane_segment import LaneMarkType, LaneSegment
from av2.map.map_api import ArgoverseStaticMap
from tqdm import tqdm
import argparse

from .visual import visual_map_gt,visual_map_gt_pivot
from .av2map_extractor import  VectorizedAV2LocalMap
from .rasterize import RasterizedLocalMap

CAM_NAMES = ['ring_front_left', 'ring_front_center', 'ring_front_right', 'ring_side_right',
    'ring_rear_right','ring_rear_left',  'ring_side_left',
     #'stereo_front_left', 'stereo_front_right',
    ]
# some fail logs as stated in av2
# https://github.com/argoverse/av2-api/blob/05b7b661b7373adb5115cf13378d344d2ee43906/src/av2/map/README.md#training-online-map-inference-models
FAIL_LOGS = [
    '0ab21841-0c08-3bae-8424-daa9b336683f',
    '0d37aee4-6508-33a2-998d-724834e80030',
    '0d9e4cff-73ff-33eb-9981-795475e62faf',
    '12c3c14b-9cf2-3434-9a5d-e0bfa332f6ce',
    '121007f3-a0cc-3795-9606-85108b800772',
    '133e2e0b-b0fe-3bb0-b1f9-c846fcfd29e8',
    '14bf638b-8f0d-35b2-a369-6d846b5b3892',
    '14c8d182-9586-3f21-ad20-c4e19ec03e2c',
    '156a412d-3699-3c1c-9ada-6ab587347996',
    '3c27dfaf-1624-39d2-9075-158824ed8e8c',
    '3b2994cb-5f82-4835-9212-0cac8fb3d164',
    '3c56f1ef-d4df-30ae-80f3-0a5b22d4d3a6',
    '3cd2847c-604e-32b4-af19-6cd0da0dcdc5',
    '3d7743c1-c0a5-3ab2-976e-84af93270f30',
    '3fa8c20e-a4b4-3af6-b9c4-6cb96f83916d',
    '4058d838-75cb-35e2-af7e-a51aaa833271',
    '40870b19-3356-3e8e-a4a4-9f34eef8ea30',
    '3f9796e9-c892-3915-b719-3292df878ece',
    '3c58172c-7a07-3ad4-bdf6-7cae60928c56',
    '3b68c074-1680-3a93-92e5-5b711406f2fe',
    '3b60751b-7a71-3a47-a743-96b96f0d9b2b',
    '1a7e18b5-d8dc-371d-be5f-03a37b113e81',
    '1bf2bf1c-64d1-308f-afd1-220de9d30290',
    '1bf2bf1c-64d1-308f-afd1-220de9d30290',
    '1d43ed4e-e705-308a-bdfa-49d99285c42a',
    '20f785b0-e11a-3757-be79-b0731286c998',
    '22dcf96c-ef5e-376b-9db5-dc9f91040f5e',
    '298715e3-b204-3bf5-b8c2-fe3be9e310e8',
    '3153b5b3-d381-3664-8f82-1d3c5ca841d2',
    '32edd7c7-8a8f-360d-bcda-83ecf431e3e6',
    '332b278a-a6b9-3bc3-b88c-241e4b03b4ef',
    '35a15c5c-fa4a-3838-a724-396e112ec95c',
    '36b38cbf-f6c5-3a12-8e7a-eb281cc9c2fc',
    '3a789fb0-5cd2-3710-b8ea-f32fce38e3ca',
    '3b2994cb-5f82-4835-9212-0cac8fb3d164',
    '3b60751b-7a71-3a47-a743-96b96f0d9b2b',
    '3b68c074-1680-3a93-92e5-5b711406f2fe',
    '3c27dfaf-1624-39d2-9075-158824ed8e8c',
    '3c51357e-f6e9-3cda-9036-fe6e6cd442fe',
    '3c56f1ef-d4df-30ae-80f3-0a5b22d4d3a6',
    '3c58172c-7a07-3ad4-bdf6-7cae60928c56',
    '3cd2847c-604e-32b4-af19-6cd0da0dcdc5',
    '3d7743c1-c0a5-3ab2-976e-84af93270f30',
    '3f9796e9-c892-3915-b719-3292df878ece',
    '3fa8c20e-a4b4-3af6-b9c4-6cb96f83916d',
    '4058d838-75cb-35e2-af7e-a51aaa833271',
    '40870b19-3356-3e8e-a4a4-9f34eef8ea30',
    '41b6f7d7-e431-3992-b783-74b9edf42215',
    '444cce44-cc82-4620-b630-1b5849284ac7',
    '45433055-2b69-3cff-8135-67b3bfa04034',
    '4619e709-c9c0-3b26-923f-23a78e231136',
    '4667e48c-4d16-38be-b277-6b0013d6588c',
    '47972731-b0ea-3c38-a10f-5ffdd42329fc',
    '4935629c-fd9e-3b2f-b68e-9489c89585df',
    '49a9df80-ab0a-31fb-9341-a79f7b0258dd',
    '49d76058-b4f0-3931-86fa-de160b4c1b88',
    '4bab74cd-aba9-4752-9e1f-006cc639d63e',
    '4f1b4bb2-b30b-3537-8fed-dd8f843f5adb',
    '50d508e2-6753-4519-a8c3-ad94a76ee948',
    '57356998-297c-330a-af4e-c6a1ad64f923',
    '5b1d8b11-4f90-3577-be0b-193e102fda82',
    '5ccb359a-2986-466c-88b2-a16f51774a8f',
    '5f5a25ff-ea07-3133-b5c6-26fada93f90f',
    '6b14d7c0-20f9-390b-af38-507a5de5998c',
    '6ee14358-31cb-3c6d-82f5-54d6a20444de',
    '6f2f7d1e-8ded-35c5-ba83-3ca906b05127',
    '71283e26-905b-3811-b9e0-c10c0253769b',
    '718a2f8d-954a-3cd8-89e6-43898cf21fee',
    '723147da-6beb-34cc-b1d7-2d4d3abc4d33',
    '71d95611-9032-3787-a66e-e26313b08d46',
    '72c31859-3676-3cbb-a773-0591d8d5799e',
    '737314f0-997a-3cd1-a652-78453bfe2b57',
    '75449af9-61a5-3a4f-95ec-3a3dc35b4cbb',
    '76038978-47aa-30ed-bfa1-2d63753a866c',
    '768cf7e2-eb6c-3468-969e-e3b0fd87b34e',
    '78cbd619-8ded-35b8-87a1-38c4f4aeb82d',
    '7a1412d3-5a53-378f-85df-ba58b2408f46',
    '7b0bf9d6-084a-31d4-9e52-d9b582a0ec84',
    '7c5e3704-33c8-3a4e-b032-9187a6f90206',
    '7ccdda39-69b1-36d1-89c8-2acc3823264b',
    '7cd08674-1787-37d9-9365-988df023724b',
    '7d1d720d-6708-3148-917a-b8dc78f1dcd9',
    '80f31501-6533-3257-9870-b0c4dbf61967',
    '8223c3d0-3b08-3889-9cdc-a88592c4bd4a',
    '84bb4b17-e7f2-3a1b-8c2b-6d6ec9a23e31',
    '85026edc-5fdf-308e-a8ef-a1aad1151c50',
    '855908a6-a848-3b7b-a4a3-bbab78a423cd',
    '855ba280-cd69-348d-9107-69e28cb8ad99',
    '88ebed6a-e9a4-3d60-9011-ec1df75cc8d6',
    '8a11791c-1d8f-3b12-bacc-38aa982b0003',
    '8aeeeeca-6a79-34ef-b667-835d53536a8f',
    '8beeb8db-28f9-396c-b752-17f906505948',
    '8ca98d88-67b5-385e-80f7-b32758668fab',
    '8d8b550e-d0be-3cbb-a371-49ec36fa619f',
    '8feb3dbe-4450-3aeb-b22b-e65128aa696b',
    '91923e20-9a05-32e0-ac53-8c09b0b60341',
    '919f13de-857f-3b1c-9f8e-7cbe500a60ae',
    '928e282f-d1a0-3e85-9582-0b33664c49e8',
    '945f3b20-778a-3581-adef-544de4a089ef',
    '953087a4-f704-37fe-a60f-82877e84a413',
    '95a47a36-1041-3924-bbd0-4dcad52c323a',
    '95acebfe-c694-3dab-9e6d-01cb501ff426',
    '982bcae9-1840-37f4-9278-3dbb63031aac',
    '98fd128c-4f32-40fc-a23c-7feb50c4478a',
    '9bdb4139-173f-33d3-8730-e29752d737d3',
    '9f6d282e-f573-31d5-80e6-9a193e80cd7d',
    'a0cb0614-ee71-3cf3-b891-a4274883362f',
    'a1c1d559-0480-39d2-94f0-1a89f0226c4f',
    'a3876690-9d49-3c98-9421-02cfe0ccb551',
    'a3e09a66-a921-3c4a-89e6-7fecf6854a3a',
    'a4f240a0-12d4-3542-a11f-0c592e90e4da',
    'a7a2236e-8f8e-34aa-9343-722f9b3bb829',
    'aa82b61f-7156-3c68-95a4-b79cebd120eb',
    'ab3d8387-8e07-37f6-a74c-cf100fb6a612',
    'ab83611b-436e-3de7-aad1-f0c9ad254196',
    'ac1b1697-42b9-4225-a666-d17f72204fa8',
    'b2a8a9aa-19cd-3ffd-b02c-0f2a47d1d0eb',
    'b403f8a3-4cad-333e-8557-d7da2e163f4b',
    'b5a1b0b0-a7fc-3a47-af82-9b25a81a8c0b',
    'b5ea60b0-2540-4efe-b60e-f421ade3c128',
    'b5f3900c-b421-3032-aef2-2e91a69d1163',
    'b66a9b8e-8fa8-3409-907f-a70ebd7051e1',
    'b6c04ab6-1c07-3e17-97d5-e870db090e52',
    'b81922e7-092f-3052-8cd1-fec6a6763295',
    'bb110668-5037-3c04-bd34-34cf1ace8d0f',
    'bbdb1e21-62eb-3230-8cef-a3b091c5edad',
    'bd4a7d9d-14e1-3c17-873d-a74d0cd6a5d7',
    'bec0f69b-832c-3898-b589-0127ddc282f3',
    'c062ba0f-7591-3225-a57d-8181622dc2da',
    'c2c0e6bc-05e5-30dd-8e5e-0e7b6106ad30',
    'c2f301b6-5d19-3296-a8ac-418ff48e052b',
    'c556f8e0-a001-3586-b2cf-d3256685c39f',
    'c67f439a-f945-33cb-8517-40c9fdf60d59',
    'c70b2c64-7a9a-37c5-b974-709fa0536675',
    'c69e348a-8e10-31dc-b71b-dd8e5cfd7211',
    'c71cd96c-8e3f-3861-9ece-fcbabebc63a8',
    'c7f5e5c1-dc52-3619-8998-420b2e280d8a',
    'ccb4e29d-e88f-3fbe-8958-67cfd62350a3',
    'cdd752d0-caee-3d95-b1db-7fc20cbbc783',
    'ce0e814a-d9df-3975-a521-d8ae9a091e96',
    'ce34ff64-0faa-3fae-a79e-985f7a5172c9',
    'cf6a99cb-b8bc-34d7-bdca-30e50e66cd74',
    'd0ba7a1b-f5ca-39d6-98d0-29c671baec65',
    'd1695c5e-08a9-44fd-8f45-93c23f700c8b',
    'd26b95e4-d200-34e2-92c9-c16fda4cd9dd',
    'd33f667d-7b6c-39aa-9ba9-eac2fa615ae1',
    'd37be0e2-8223-3eeb-a0e2-c4b75d5ff87b',
    'd3dc783e-663a-31b1-bd85-46e04ca693db',
    'd3efe9ba-f10a-35e7-b17e-6850c66693fe',
    'd551b8f9-feab-3946-9524-219e07988341',
    'd58d55ea-f30c-3622-8303-1574616b9865',
    'd6ba4898-1369-3521-981c-b9ac57420418',
    'd78b78a0-2322-32c2-833a-e42ddc132d30',
    'dc4d148d-f84c-307c-b2b7-f0cd7c267f57',
    'dd251cc5-736d-3b76-8ad3-3f6cb138178e',
    'df1935dc-1e5f-3f4d-bdcb-e6c2bcb07667',
    'df5d0b0e-5bcb-304a-a167-18b92d0f1d45',
    'e0ba7664-d287-39df-8193-00d60cae1417',
    'e10475f7-0d56-3a75-870d-d4206fa165d7',
    'e125bb91-dcaf-3013-9cc7-da653d7e11e1',
    'e1e9d341-716f-3613-9ec2-2201c72361af',
    'e2043284-6122-3cc9-a7e7-f091a16361b7',
    'e424d4f7-4b28-322f-b630-31d42ae528eb',
    'e66d1403-755b-3f63-938b-a2a69446a48a',
    'e7e178aa-931a-4674-9bff-9278a54e6aae',
    'eb142141-683a-3a6d-a207-0302b1ff260d',
    'eb222d5d-0052-3ce7-9b87-19e09054a2c0',
    'ede387f4-f390-3f0e-a071-eb543b73ed73',
    'eed8593d-60e3-3e41-9fea-55f544b01749',
    'ef4a46c4-138e-3478-b94e-3e60a567ec7d',
    'f110598d-7e01-3ed7-a227-4e958987a31f',
    'f3d1e3c3-2770-3504-a592-b62619598812',
    'f3f8f680-e471-3662-a06a-0c00e6d88f43',
    'f41d0e8f-856e-3f7d-a3f9-ff5ba7c8e06d',
    'f46707f9-435f-3a06-9017-deae11feab53',
    'f4c6ade0-7b9e-4ad7-8d86-13d2f4c91499',
    'f4cb6ba4-cd0f-30cc-9cc9-52bd14bfb3cc',
    'f4d1a3c3-5002-336b-a67f-775b3725237e',
    'f61bcee1-2964-3c4b-95a5-697df5f42f47',
    'f6350a4f-eee8-31bd-8520-28f9c81c98a8',
    'f7cf93d8-f7bd-3799-8500-fbe842a96f63',
    '022af476-9937-3e70-be52-f65420d52703',
    '3c51357e-f6e9-3cda-9036-fe6e6cd442fe',
    '36b38cbf-f6c5-3a12-8e7a-eb281cc9c2fc',
    '35a15c5c-fa4a-3838-a724-396e112ec95c',
    '332b278a-a6b9-3bc3-b88c-241e4b03b4ef',
    '32edd7c7-8a8f-360d-bcda-83ecf431e3e6',
    '3153b5b3-d381-3664-8f82-1d3c5ca841d2',
    '2b044433-ddc1-3580-b560-d46474934089',
    '22dcf96c-ef5e-376b-9db5-dc9f91040f5e',
    '20f785b0-e11a-3757-be79-b0731286c998',
    '1d43ed4e-e705-308a-bdfa-49d99285c42a',
    '3b60751b-7a71-3a47-a743-96b96f0d9b2b',
    '40870b19-3356-3e8e-a4a4-9f34eef8ea30',
    'de586ff4-3413-367d-befc-ad022b73592b',
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
    '01bb304d-7bd8-35f8-bbef-7086b688e35e',
    '453e5558-6363-38e3-bf9b-42b5ba0a6f1d',
    '96dd6923-994c-3afe-9830-b15bdfd60f64',
    '3503b283-fbcd-3835-8779-0cb2b7ef55b0', #test dataset
    '0f0cdd79-bc6c-35cd-9d99-7ae2fc7e165c',
    '1ad57a00-cc61-3f5f-9e2a-9981a57e9856',
    '2a930061-3d8c-3915-8aac-f81199db95d8',
    '2e95b33b-8ea1-3b48-875b-2f35f3092059',
    '2ee0eda7-151a-3957-bab5-1e5370192122',
    '32835bfa-e53b-3526-9ec0-b0efcd11cbdf',
    '34c79495-dbdf-393d-bcc6-e6f92f797628',
    '4b6fd841-5e08-30cd-b61b-d51aa36d71dd',
    '4bf8e9ff-e1a1-3a22-a9d1-80f3846c0263',
    '4e6d6bcd-8718-3e71-b9c1-7c352c991a56',
    '5133fc11-ecb7-318d-8338-6502d2390f24',
    '557dd6a4-2b80-3264-9c13-f70094526174',
    '65f1eefa-cbc3-3d53-9991-dc0500ae9183',
    '803c44cc-e1de-3797-9b5f-15324a1604f8',
    '84ed050c-635f-36ec-9c28-8a0c10f5cf11',
    '9282db22-c361-3456-a7b5-414959f5f25e',
    '91ac892f-d2c1-3143-b5c5-f0d4640cfc0d',
    'b0116f1c-f88f-3c09-b4bf-fc3c8ebeda56',
    'b4c0dac8-09d8-3f4b-be7b-9f89473c250c',
    'b7cbdba9-18ac-393a-8352-4841ffee722e',
    'b869e2cb-3804-389a-a3c7-b80f57c7d2ac',
    'c6b7a5fb-8cd8-3ee2-8e99-b788eb02e731',
    'd4c7aa45-dfd6-3d71-bb8a-40efd5110d3b',
    'd67d020a-4d28-3bfd-891d-d6aa7dcf0a69',
    'e4221cc6-a19d-31ca-bf94-031adb0ea390',
    'e858fb96-6b1f-3025-b40a-f71fd8d28c32',
    'e95c8cc2-ddb3-3e7b-b8c3-e7584a778464',
    'f6107596-76e0-3064-a4a6-86332a90e539',
    'f7d568d4-0836-3f47-b330-f8d204c4b96e',
    'fee0f78c-cf00-35c5-975b-72724f53fd64',
    'e596b305-c951-3081-ae02-85406a473840',
    'c0e93b69-158e-3f05-931b-999bdf9d753a',
    'b2d9d8a5-847b-3c3b-aed1-c414319d20af',
    '4d73c4eb-5de9-300c-b34f-ff5d0af89653',
    '131bd3d9-4f85-3ba3-b569-eb88308d79d5',
    '1bd7db3a-0b42-31cf-ac1a-de88fd9fa721',
    '6f3dbf4b-9559-340c-a3e4-cbe655bf2059',
    'c45888cf-30f5-3e27-abeb-4f55caecc1f0',
    '67be173f-28a9-3bcc-b110-4b81dfe3bf5e',
    'a674e2e5-3dfd-3dd5-8503-192357b0e96c',
    '45488531-3648-3e2d-8f9c-3c287032112d',
    'a89557fc-1268-36e5-9cce-335f2da27bc8',
    '6784f175-e69d-3802-99df-d21ec2081878',
    'a674e2e5-3dfd-3dd5-8503-192357b0e96c',
    '9a82e3c8-1738-3f85-9245-1d3717171d2f',
    'c19d4b20-814d-3b2e-94e5-5d3003631496',
    '6da5d01e-54a7-3d7a-b86b-e0d6f8d3971d',
    'c9fc62c5-a289-36e3-a900-7e7807eb2716',
    'f77889f6-ef5a-4eed-a4cd-5d67d4a6e9c5',
    'a7f532a3-87de-3129-8864-258396fd0b50',
    'b42dc943-8b33-3b79-a260-14eb9f58a991',
    '399064b4-6df3-3de8-8793-2738f8723ee3',
    'fb207d3b-d2d5-3100-94c0-9145aebc770b',
    '386c34fc-ff56-371c-9288-6ba42620f23a',
    'e95e20d1-7f04-34b9-9105-4333f11bf6b9',
    'c42d34f3-78d5-35be-9c47-77d297caebfe',
    'a4400a38-bc38-391c-b102-ba385d7e475e',
    'd8192bbb-3b00-3c68-a79a-65872ea4276f',
    'b48a15fb-2e84-34df-946f-ad72b3d7296f',
    'f554d503-4901-3b97-9516-a16398c66631',
    'a89557fc-1268-36e5-9cce-335f2da27bc8',
    '412ccada-28df-3de2-b394-9cba3fca5bdf',
    'de23dfe1-c0b1-441b-810b-324090dc171b',
    '6a6e93f0-a130-3340-975b-b2c88b16d343',
    '1c8648f9-e7a1-3056-a2c0-19c8827a6a50',
    'a1358c59-b28d-3ddb-af1c-3a5d1c394ef5',
    '7b7f86ca-b430-3872-a131-ff5b4a6b5dcf',
    '87e61f5a-083c-305e-9ff4-5f699e85900a',
    '44e49f1b-17e1-3da6-9b8b-97e754d58f7a',
    '9a82e3c8-1738-3f85-9245-1d3717171d2f',
    '51428934-b0a7-3507-94e3-31d37bba38a3',
    'd70660da-4250-3ad1-a2d0-6a2d97b5379f',
    'f849731b-d288-3bec-8f35-6bea979f7dd8',
    'cf79d751-5d2a-3d5c-96a2-bb8d603f21e0',
    '9d16e76e-46ae-38c6-8399-99218514afde',
    '613558a1-6a8e-3fda-8fa6-1045a064a0f9',
    '9da07440-1001-3b00-a29f-c8bdc2f2b7d4',
    '0c6e62d7-bdfa-3061-8d3d-03b13aa21f68',
    'bc20a6d3-2db2-3849-8843-1e1b8c93e5db',
    'd70bc0a2-9d7f-36bd-bf37-ed798b10b71b',
    '6626b7b2-bcc8-4497-ae92-307ceacd5010',
    '4fcdebe7-b52f-39e7-a5bc-c664eeba5e7b',
    '1ca5291b-3178-3a93-a117-001497899b79',
    'bc073372-a582-4c57-a579-a7fcf15b49de',
    '7c4e5ad1-d604-3e44-81ae-68f7bfe21d27',
    '4612f4a4-59b0-37d4-b3d1-400a1324920c',
    'f7c4cf87-6bab-3723-bd74-1c9ac5add9cb',
]

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        help='specify the root path of dataset')
    parser.add_argument(
        '--nproc',
        type=int,
        default=64,
        required=False,
        help='workers to process data')
    args = parser.parse_args()
    return args

def create_av2_infos_mp(root_path,
                        info_prefix,
                        dest_path=None,
                        split='train',
                        num_multithread=64):
    """Create info file of av2 dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        dest_path (str): Path to store generated file, default to root_path
        split (str): Split of the data.
            Default: 'train'
    """
    root_path = osp.join(root_path, split)
    if dest_path is None:
        dest_path = root_path
    
    loader = AV2SensorDataLoader(Path(root_path), Path(root_path))
    log_ids = list(loader.get_log_ids())
    # import pdb;pdb.set_trace()
    for l in FAIL_LOGS:
        if l in log_ids:
            log_ids.remove(l)

    print('collecting samples...')
    start_time = time.time()
    print('num cpu:', multiprocessing.cpu_count())
    print(f'using {num_multithread} threads')

    # to supress logging from av2.utils.synchronization_database
    sdb_logger = logging.getLogger('av2.utils.synchronization_database')
    prev_level = sdb_logger.level
    sdb_logger.setLevel(logging.CRITICAL)

    # FIXME: need to check the order
    pool = Pool(num_multithread)
    fn = partial(get_data_from_logid, loader=loader, data_root=root_path)
    rt = pool.map_async(fn, log_ids)
    pool.close()
    pool.join()
    results = rt.get()

    samples = []
    discarded = 0
    sample_idx = 0
  
    for _samples, _discarded in results:
        for i in range(len(_samples)):
            _samples[i]['sample_idx'] = sample_idx
            sample_idx += 1
        samples += _samples
        discarded += _discarded

    
    sdb_logger.setLevel(prev_level)
    print(f'{len(samples)} available samples, {discarded} samples discarded')

    print('collected in {}s'.format(time.time()-start_time))
    infos = dict(samples=samples)

    return infos
    # mmcv.dump(samples, info_path)

def get_divider(avm):
    divider_list = []
    for ls in avm.get_scenario_lane_segments():
            for bound_type, bound_city in zip([ls.left_mark_type, ls.right_mark_type], [ls.left_lane_boundary, ls.right_lane_boundary]):
                if bound_type not in [LaneMarkType.NONE,]:
                    divider_list.append(bound_city.xyz)
    return divider_list

def get_boundary(avm):
    boundary_list = []
    for da in avm.get_scenario_vector_drivable_areas():
        boundary_list.append(da.xyz)
    return boundary_list

def get_ped(avm):
    ped_list = []
    for pc in avm.get_scenario_ped_crossings():
        ped_list.append(pc.polygon)
    return ped_list

def get_data_from_logid(log_id, loader: AV2SensorDataLoader, data_root):
    samples = []
    discarded = 0
   
    log_map_dirpath = Path(osp.join(data_root, log_id, "map"))
    vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
    # vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
    if not len(vector_data_fnames) == 1:
        raise RuntimeError(f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
    vector_data_fname = vector_data_fnames[0]
    vector_data_json_path = vector_data_fname
    avm = ArgoverseStaticMap.from_json(vector_data_json_path)
    # import pdb;pdb.set_trace()
    
    # We use lidar timestamps to query all sensors.
    # The frequency is 10Hz
    cam_timestamps = loader._sdb.per_log_lidar_timestamps_index[log_id]
    for ts in [cam_timestamps[0]]:
        cam_ring_fpath = [loader.get_closest_img_fpath(
                log_id, cam_name, ts
            ) for cam_name in CAM_NAMES]
        lidar_fpath = loader.get_closest_lidar_fpath(log_id, ts)

        # If bad sensor synchronization, discard the sample
        if None in cam_ring_fpath or lidar_fpath is None:
            discarded += 1
            continue

        cams = {}
        for i, cam_name in enumerate(CAM_NAMES):
            pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)
            cam_timestamp_ns = int(cam_ring_fpath[i].stem)
            cam_city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
            cams[cam_name] = dict(
                img_fpath=str(cam_ring_fpath[i]),
                intrinsics=pinhole_cam.intrinsics.K,
                extrinsics=pinhole_cam.extrinsics,
                e2g_translation=cam_city_SE3_ego.translation,
                e2g_rotation=cam_city_SE3_ego.rotation,
            )
        
        city_SE3_ego = loader.get_city_SE3_ego(log_id, int(ts))
        e2g_translation = city_SE3_ego.translation
        e2g_rotation = city_SE3_ego.rotation
        
        
        map_elements = {}
        map_elements['divider'] = get_divider(avm)
        map_elements['ped_crossing'] = get_ped(avm)
        map_elements['boundary'] = get_boundary(avm)
    
        samples.append(dict(
            e2g_translation=e2g_translation,
            e2g_rotation=e2g_rotation,
            cams=cams, 
            lidar_fpath=str(lidar_fpath),
            # map_fpath=map_fname,
            timestamp=ts,
            log_id=log_id,
            token=str(log_id+'_'+str(ts)),
            annotation = map_elements
            ))

        
    return samples, discarded


if __name__ == '__main__':
    args = parse_args()
    for name in ['train', 'val', 'test']:
        create_av2_infos_mp(
            root_path=args.data_root,
            split=name,
            info_prefix='av2',
            dest_path=args.data_root,)