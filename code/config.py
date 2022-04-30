
class Config:

    WIDTH = 128
    HEIGHT = 128
    STATE_CHANNEL = 2
    ENTROPY_BETA = 0.01
    SEED = 100
    USE_SEG = True

    # train
    MAX_GLOBAL_EP = 100001
    MAX_EP_STEPS = 20
    EP_BOOTSTRAP = 100
    UPDATE_GLOBAL_ITER = 8
    SAMPLE_BATCH_SIZE = 4
    FREQUENCY_PLANNER = 2
    FREQUENCY_VAE = 1

    GPU_ID = 0
    NF = 16
    BOTTLE = 64
    SCORE_THRESHOLD = 0.98
    WARMUP_EPS = 100
    MEMORY_SIZE = 800
    TAU = 0.005
    FIXED_ALPHA = 0.01
    GAMMA = 0.99
    INIT_TEMPERATURE = 0.1
    BSPLINE_AUG = False

    PRE_TRAINED = False

    # main parameters
    LEARNING_RATE = 4e-5

    # server
    IMAGE_TYPE = 'liver' # liver, brain
    DATA_TYPE = 'lits' # liver, brain
    TRAIN_DATA = '/datasets/affined_3d/train_3d.h5'
    TEST_DATA = '/datasets/liver/lspig_affine_test_3d.h5'
    # TEST_DATA = '/datasets/liver/sliver_affine_test_3d.h5'
    TRAIN_SEG_DATA = '/datasets/affined_3d/affine_test_3d.h5'
    ATLAS = '/datasets/affined_3d/atlas.npz'

    LOG_DIR = './log/'
    MODEL_PATH = './model/'
    PROCESS_PATH = 'process'
    TEST_PATH = 'result'
    ACTOR_MODEL = MODEL_PATH + 'actor.ckpt'
    ENCODER_MODEL = MODEL_PATH + 'encoder.ckpt'


    # for evaluation
    idx = 10000
    ACTOR_MODEL = MODEL_PATH + 'planner_{}.ckpt'.format(idx)
    CRITIC1_MODEL = MODEL_PATH + 'critic1_{}.ckpt'.format(idx)
    CRITIC2_MODEL = MODEL_PATH + 'critic2_{}.ckpt'.format(idx)
    ACTOR_MODEL_RL = MODEL_PATH + 'actor_{}.ckpt'.format(idx)



