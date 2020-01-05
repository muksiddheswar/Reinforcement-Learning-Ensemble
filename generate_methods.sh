echo > methods.md

for md in .2.methods.Qlearning.md .2.methods.sarsa.md .2.methods.actor-critic.md .2.methods.QVlearning.md .2.methods.ACLA.md .2.methods.majority-voting.md .2.methods.rank-voting.md .2.methods.boltzmann-multiplication.md .2.methods.boltzmann-addition.md .2.methods.experiments.md
do
    cat ${md} >> methods.md
    echo >> methods.md
    echo >> methods.md
done