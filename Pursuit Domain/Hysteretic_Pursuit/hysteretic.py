

#論文的內容
def hysteretic(table, states, action, alpha, beta, r, gamma, new_states):
    
    maximum = 0 if not table[new_states] else max(table[new_states].values())

    delta = r + gamma * maximum - table[states][action]
    if delta >= 0:
        table[states][action] += delta * alpha
    else:
        table[states][action] += delta * beta
            
    return table
