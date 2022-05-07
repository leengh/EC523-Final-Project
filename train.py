from DQN_agent import DQN_Agent

# =====================================================
#               BINARY REWARDS
# =====================================================

agent = DQN_Agent(load_weights_path=None,
                           learning_rate=0.0005, train=True, model_type="binary")

# =====================================================
#               CATEGORICAL REWARDS
# =====================================================

# agent = DQN_Agent(load_weights_path=None,
#                     learning_rate=0.0005, train=True, model_type="categorical")

# =====================================================
#               EUCLIDEAN REWARDS
# =====================================================

# agent = DQN_Agent(load_weights_path=None,
#                     learning_rate=0.0005, train=True, model_type="euclidean")

# =====================================================
#               MANHATTAN REWARDS
# =====================================================

# agent = DQN_Agent(load_weights_path=None,
#                               learning_rate=0.0005, train=True, model_type="manhattan")

# =====================================================
#                       TRAIN
# =====================================================

agent.train()