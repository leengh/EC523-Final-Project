from DQN_agent import DQN_Agent

# =====================================================
#               BINARY REWARDS
# =====================================================

agent = DQN_Agent(load_weights_path="model_weights/1000-episodes-binary.pt",
                    train=False, model_type="binary")

# =====================================================
#               CATEGORICAL REWARDS
# =====================================================

# agent = DQN_Agent(load_weights_path="model_weights/1000-episodes-categorical.pt",
#                      train=False, model_type="categorical")

# =====================================================
#               EUCLIDEAN REWARDS
# =====================================================

# agent = DQN_Agent(load_weights_path="model_weights/1000-episodes-euclidean.pt",
#                      train=False, model_type="euclidean")

# =====================================================
#               MANHATTAN REWARDS
# =====================================================

# agent = DQN_Agent(load_weights_path="model_weights/1000-episodes-manhattan.pt",
#                      train=False, model_type="manhattan")


agent.test()
