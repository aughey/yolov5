const express = require('express')
const app = express()
const port = 5000
const redis = require("redis");
const client = redis.createClient({ 'return_buffers': true });
var bodyParser = require('body-parser');

var mqtt = require('mqtt')
var mqttclient = mqtt.connect('mqtt://localhost')



app.use(bodyParser.raw({
    type: 'application/octet-stream',
    limit: '10mb'
}));




client.on("error", function (error) {
    console.error(error);
});


app.get('/', (req, res) => {
  res.send('Hello World!')
})

app.post("/set/:key", (req, res) => {
    const key = req.params.key
    const value = req.body;
    client.setex(key, 60, value);
    res.send("ok");
})

app.get("/get/:key.:ext", (req, res) => {
    const key = req.params.key
    const ext = req.params.ext

    res.contentType("foo." + ext)

    client.get(key, (err,value) => {
        res.send(value)
    })
})

app.post("/message/:channel", (req, res) => {
    const channel = req.params.channel
    const body = req.body;
  
    mqttclient.publish(channel, body)
    res.send("ok")
})

app.get('/testmessage/:channel/:data', (req, res) => {
    const channel = req.params.channel
    const data = req.params.data;
    mqttclient.publish(channel, data)
    res.send("ok")
})

app.get('/testset/:key/:data', (req, res) => {
    const key = req.params.key
    const data = req.params.data;

    client.set(key, data, redis.print);
    res.send("ok");
})

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
})// JavaScript source code
