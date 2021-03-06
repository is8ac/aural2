// Package tftrain runs training graphs
package tftrain

import (
	"bytes"
	"errors"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"strings"

	"github.com/golang/protobuf/proto"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	pbtf "github.ibm.com/Blue-Horizon/aural2/tfutils/demo/protobuf/tensorflow/core/framework"
)

var logger = log.New(os.Stdout, "tftrain: ", log.Lshortfile)

func loadTrainGraph(path string) (graph *tf.Graph, err error) {
	graphBytes, err := ioutil.ReadFile(path)
	if err != nil {
		return
	}
	graph = tf.NewGraph() // make a new graph,
	// and import the graphdef into it.
	if err = graph.Import(graphBytes, ""); err != nil {
		logger.Println(err)
		return
	}
	return
}

func getOP(graph *tf.Graph, name string) (operation *tf.Operation, err error) {
	operation = graph.Operation(name)
	if operation == nil {
		err = errors.New("can't find operation " + name)
		return
	}
	return
}

// convert a tf.Graph to a proto unmarshalled graphDef
func tfGraphToPbGraph(tfGraph *tf.Graph) (pbGraph *pbtf.GraphDef, err error) {
	buff := bytes.Buffer{}
	_, err = tfGraph.WriteTo(&buff)
	if err != nil {
		return
	}
	pbGraph = &pbtf.GraphDef{}
	err = proto.Unmarshal(buff.Bytes(), pbGraph)
	if err != nil {
		logger.Println(err)
	}
	return
}

// convert a pbtf.GraphDef to a tf.Graph
func pbGraphToTfGraph(pbtfGraph *pbtf.GraphDef) (tfGraph *tf.Graph, err error) {
	pbGraphBytes, err := proto.Marshal(pbtfGraph)
	if err != nil {
		return
	}
	tfGraph = tf.NewGraph()
	err = tfGraph.Import(pbGraphBytes, "")
	return
}

// getDeps for the headName
func getDeps(allNodes map[string]*pbtf.NodeDef, nodeName string) (requiredNodes []*pbtf.NodeDef) {
	suffix := regexp.MustCompile(`\:[0-9]$`)             // matches the output at the end of the operation name
	outputSuffix := suffix.FindString(nodeName)          // get the output suffix
	nodeName = strings.TrimRight(nodeName, outputSuffix) // remove it, so as to get only the operation name
	if nodeName[0] == '^' {                              // strip leading ^ from name.
		nodeName = nodeName[1:] // I don't know why TF adds it sometimes, but it's not needed AFACT.
	}
	node, prs := allNodes[nodeName]
	if prs { // if the operation is not present, ignore it, it was probably already included.
		requiredNodes = []*pbtf.NodeDef{node}  // start the list of nodes with the head
		delete(allNodes, node.Name)            // remove the head
		for _, inputName := range node.Input { // for each of the nodes inputs,
			requiredNodes = append(requiredNodes, getDeps(allNodes, inputName)...) // add the node and its inputs.
		}
	}
	return
}

// convert list of nodes to a map of nodeNames to nodes
func nodesToMap(nodeList []*pbtf.NodeDef) (nodeMap map[string]*pbtf.NodeDef) {
	nodeMap = map[string]*pbtf.NodeDef{}
	for _, node := range nodeList {
		nodeMap[node.Name] = node
	}
	return
}

func listVarNames(nodes []*pbtf.NodeDef) (varNames []string) {
	for _, node := range nodes {
		if node.Op == "VariableV2" {
			varNames = append(varNames, node.Name)
		}
	}
	return
}

func listVarHandleNames(nodes []*pbtf.NodeDef) (varHandleNames []string) {
	for _, varNode := range nodes {
		if varNode.Op == "VariableV2" {
			for _, node := range nodes {
				if len(node.Input) == 1 && node.Input[0] == varNode.Name && node.Op == "Identity" {
					varHandleNames = append(varHandleNames, node.Name)
				}
			}
		}
	}
	return
}

func getInitNames(nodesMap map[string]*pbtf.NodeDef, initOPname string) (varNames, initValueNames []string) {
	for _, initNodeName := range nodesMap[initOPname].Input {
		if initNodeName[0] == '^' { // strip leading ^ from name.
			initNodeName = initNodeName[1:] // I don't know why TF adds it sometimes, but it's not needed AFACT.
		}
		logger.Println(initNodeName)
		initNode := nodesMap[initNodeName]
		logger.Println(initNode.Input)
		if len(initNode.Input) != 2 {
			panic(initNode.Name + "does not have 2 inputs")
		}
		varNames = append(varNames, initNode.Input[0])
		initValueNames = append(initValueNames, initNode.Input[1])
	}
	return
}

// pull out the actual values of each var as tensors.
func evalVars(varNames []string, graph *tf.Graph, sess *tf.Session) (tensors []*tf.Tensor, err error) {
	varOutputs := make([]tf.Output, len(varNames))
	for i, name := range varNames { // for each variable name,
		op := graph.Operation(name) // get it from the graph.
		if op == nil {              // if it wasn't in the graph,
			err = errors.New("can't find node " + name) // complain
			return
		}
		//logger.Println(op.Output(0).Shape(), op.Name())
		varOutputs[i] = op.Output(0) // if not, put the operations first output in the list of outputs
	}
	tensors, err = sess.Run( // run graph, pulling on all the vars.
		map[tf.Output]*tf.Tensor{},
		varOutputs,
		[]*tf.Operation{},
	)
	return
}

// addConstVarInitializer adds a constant of the value tensor to the graph
func addConstVarInitializer(graph *tf.Graph, variable tf.Output, tensor *tf.Tensor, name string) (assignOP *tf.Operation, err error) {
	constOp, err := graph.AddOperation(tf.OpSpec{
		Name: name,
		Type: "Const",
		Attrs: map[string]interface{}{
			"dtype": tensor.DataType(),
			"value": tensor,
		}})
	if err != nil {
		return
	}
	opspec := tf.OpSpec{
		Type: "AssignVariableOp",
		Input: []tf.Input{
			variable,
			constOp.Output(0),
		},
	}
	assignOP, err = graph.AddOperation(opspec)
	return
}

// addConst adds a constant of the value tensor to the graph
func addConst(graph *tf.Graph, tensor *tf.Tensor, name string) (err error) {
	_, err = graph.AddOperation(tf.OpSpec{
		Name: name,
		Type: "Const",
		Attrs: map[string]interface{}{
			"dtype": tensor.DataType(),
			"value": tensor,
		}})
	if err != nil {
		return
	}
	return
}

// Freeze replaces all variables with consts, and strip off non required nodes
func Freeze(tfGraph *tf.Graph, sess *tf.Session, headNames []string) (frozenTfGraph *tf.Graph, err error) {
	// convert to pb graph once so we can list all vars
	pbGraph, err := tfGraphToPbGraph(tfGraph)
	if err != nil {
		logger.Println(err)
		return
	}
	varNames := listVarNames(pbGraph.Node)            // extract the names of the vars
	tensors, err := evalVars(varNames, tfGraph, sess) // evaluate the vars to get their actual values
	if err != nil {
		logger.Println(err)
		return
	}
	for i, name := range varNames { // for each name,
		addConst(tfGraph, tensors[i], "frozen/"+name) // add a constant to the tf.Graph in the frozen/ namespace
	}
	// convert to pb graph again now that we have included the consts.
	pbGraph, err = tfGraphToPbGraph(tfGraph)
	if err != nil {
		logger.Println(err)
		return
	}
	nodesMap := nodesToMap(pbGraph.Node)
	for _, name := range varNames { // for each variable,
		constNode := nodesMap["frozen/"+name] // get the frozen constant of its value
		constNode.Name = name                 // give it the same name as the variable,
		nodesMap[name] = constNode            // replace the variable with the constant
	}
	var deps []*pbtf.NodeDef
	for _, headName := range headNames {
		deps = append(deps, getDeps(nodesMap, headName)...) // extract the nodes that the output node depends on.
	}
	pbGraph.Node = deps                            // replace the nodes of the pb graph with the shorter list of required nodes.
	frozenTfGraph, err = pbGraphToTfGraph(pbGraph) // convert back to a tf.Graph so go tf will do checking for us.
	if err != nil {
		logger.Println(err)
	}
	return
}

// TrainableFreeze replaces all the assign consts for each variables with the trained value, and strip of non required nodes.
// The resulting graph will still require initialisation to use, and will be larger, but can be used just like a python genorated train graph.
func TrainableFreeze(tfGraph *tf.Graph, sess *tf.Session, headNames []string) (frozenTfGraph *tf.Graph, err error) {
	// convert to pb graph once so we can list all vars
	pbGraph, err := tfGraphToPbGraph(tfGraph)
	if err != nil {
		logger.Println(err)
		return
	}
	nodesMap := nodesToMap(pbGraph.Node)              // convert the list of names to a map.
	varNames := listVarNames(pbGraph.Node)            // extract the names of all the vars
	tensors, err := evalVars(varNames, tfGraph, sess) // evaluate the vars to get their actual values
	if err != nil {
		logger.Println(err)
		return
	}
	var deps []*pbtf.NodeDef
	for _, headName := range headNames {
		deps = append(deps, getDeps(nodesMap, headName)...) // extract the nodes that the output node depends on.
	}
	//logger.Println("deps:", deps)
	pbGraph.Node = deps                            // replace the nodes of the pb graph with the shorter list of required nodes.
	frozenTfGraph, err = pbGraphToTfGraph(pbGraph) // convert back to a tf.Graph

	initOutputs := make([]tf.Output, len(varNames))
	for i, varName := range varNames { // for each name,
		constOp, err := frozenTfGraph.AddOperation(tf.OpSpec{
			Name: "frozen/" + varName,
			Type: "Const",
			Attrs: map[string]interface{}{
				"dtype": tensors[i].DataType(),
				"value": tensors[i],
			}})
		if err != nil {
			panic(err)
		}
		variable := frozenTfGraph.Operation(varName)
		if variable == nil {
			panic("can't find " + varName)
		}
		initOP, err := frozenTfGraph.AddOperation(tf.OpSpec{
			Name: "varinit/" + varName,
			Type: "Assign",
			Input: []tf.Input{
				variable.Output(0),
				constOp.Output(0),
			},
		})
		if err != nil {
			panic(err)
		}
		initOutputs[i] = initOP.Output(0)
	}
	initOP, err := frozenTfGraph.AddOperation(tf.OpSpec{
		Name:  "init",
		Type:  "ShapeN",
		Input: []tf.Input{tf.OutputList(initOutputs)},
	})
	if err != nil {
		panic(err)
	}
	logger.Println(initOP.Name(), initOP.Type())
	return
}

// BatchTrain trains a model
func BatchTrain(
	graph *tf.Graph, // the tf compute graph containing OPs for training.
	numBatches int, // how many mini batches to run
	inputName, targetName, trainOpName, initOpName, lossOpName string, // op names for the input, target, train and loss OPs.
	outputOpNames []string, // the names of the outputs to be preserved when freezing.
	lossHandlerFunction func(int, float32), // lossHandlerFunction will be called each miniBatch
	getTrainingData func(int) (*tf.Tensor, *tf.Tensor, error), // getTrainingData is called to get the input and target tensors for each miniBatch
) (frozen *tf.Graph, err error) { // returns a frozen graph.
	initOP, err := getOP(graph, initOpName)
	if err != nil {
		return
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		return
	}
	_, err = sess.Run(map[tf.Output]*tf.Tensor{}, []tf.Output{}, []*tf.Operation{initOP}) // initialize the vars.
	if err != nil {
		return
	}
	trainOP, err := getOP(graph, trainOpName)
	if err != nil {
		return
	}
	lossOP, err := getOP(graph, lossOpName)
	if err != nil {
		return
	}
	inputsPH, err := getOP(graph, inputName)
	if err != nil {
		return
	}
	targetsPH, err := getOP(graph, targetName)
	if err != nil {
		return
	}
	var inputTensor *tf.Tensor
	var targetTensor *tf.Tensor
	var results []*tf.Tensor
	for i := 0; i < numBatches; i++ { // for each miniBatch,
		inputTensor, targetTensor, err = getTrainingData(i) // get the training data
		if err != nil {
			return
		}
		results, err = sess.Run(
			map[tf.Output]*tf.Tensor{inputsPH.Output(0): inputTensor, targetsPH.Output(0): targetTensor}, // feed it the inputs and outputs
			[]tf.Output{lossOP.Output(0)},                                                                // evaluate loss
			[]*tf.Operation{trainOP},                                                                     // pull on the train OP, but don't try to get its output. (It doesn't have one.)
		)
		if err != nil {
			logger.Println(err)
			return
		}
		go lossHandlerFunction(i, results[0].Value().(float32)) // give the loss function the loss.
	}
	frozen, err = Freeze(graph, sess, outputOpNames) // convert the perishable vars to constants which will persist in the graph itself.
	sess.Close()
	return
}

// NewOnlineSess makes a new OnlineSess
func NewOnlineSess(
	graph *tf.Graph,
	inputName, targetName, trainOpName, initOpName, lossOpName, outputOpName, inferInputName string,
	outputOpNames []string,
) (oSess OnlineSess, err error) {
	initOP, err := getOP(graph, initOpName)
	if err != nil {
		return
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		return
	}
	_, err = sess.Run(map[tf.Output]*tf.Tensor{}, []tf.Output{}, []*tf.Operation{initOP}) // init the vars
	if err != nil {
		return
	}
	trainOP, err := getOP(graph, trainOpName)
	if err != nil {
		return
	}
	lossOP, err := getOP(graph, lossOpName)
	if err != nil {
		return
	}
	inputsPH, err := getOP(graph, inputName)
	if err != nil {
		return
	}
	inferInputPH, err := getOP(graph, inferInputName)
	if err != nil {
		return
	}
	targetsPH, err := getOP(graph, targetName)
	if err != nil {
		return
	}
	outputOP, err := getOP(graph, outputOpName)
	if err != nil {
		return
	}
	oSess = OnlineSess{
		Graph:         graph,
		Sess:          sess,
		trainInputPH:  inputsPH.Output(0),
		inferInputPH:  inferInputPH.Output(0),
		targetPH:      targetsPH.Output(0),
		trainOP:       trainOP,
		loss:          lossOP.Output(0),
		output:        outputOP.Output(0),
		initOPName:    initOP.Name(),
		outputOPnames: outputOpNames,
	}
	return
}

// OnlineSess stores a model in the process of training.
type OnlineSess struct {
	Graph         *tf.Graph
	Sess          *tf.Session
	trainInputPH  tf.Output
	inferInputPH  tf.Output
	targetPH      tf.Output
	trainOP       *tf.Operation
	loss          tf.Output
	output        tf.Output
	initOPName    string
	outputOPnames []string
}

// Train trains one mini batch
func (oSess OnlineSess) Train(inputTensor *tf.Tensor, targetTensor *tf.Tensor) (loss float32, err error) {
	results, err := oSess.Sess.Run(
		map[tf.Output]*tf.Tensor{oSess.trainInputPH: inputTensor, oSess.targetPH: targetTensor},
		[]tf.Output{oSess.loss},
		[]*tf.Operation{oSess.trainOP},
	)
	if err != nil {
		return
	}
	loss = results[0].Value().(float32)
	return
}

// Save writes the variable values to their initialisers.
func (oSess OnlineSess) Save() (graph *tf.Graph, err error) {
	headNames := append(oSess.outputOPnames,
		oSess.trainOP.Name(),
		//oSess.initOPName,
		oSess.loss.Op.Name(),
		oSess.inferInputPH.Op.Name(),
		oSess.output.Op.Name(),
	)
	graph, err = TrainableFreeze(oSess.Graph, oSess.Sess, headNames)
	return
}

// Infer runs the graph
func (oSess OnlineSess) Infer(inputTensor *tf.Tensor) (outputTensor *tf.Tensor, err error) {
	results, err := oSess.Sess.Run(
		map[tf.Output]*tf.Tensor{oSess.inferInputPH: inputTensor},
		[]tf.Output{oSess.output},
		nil,
	)
	if err != nil {
		return
	}
	outputTensor = results[0]
	return
}

// Run the graph. Just a wrapper around tf.Session.Run() For simple cases you probably want .Infer()
func (oSess OnlineSess) Run(feeds map[tf.Output]*tf.Tensor, fetches []tf.Output) (results []*tf.Tensor, err error) {
	results, err = oSess.Sess.Run(feeds, fetches, nil)
	return
}
